#!/usr/bin/env python3
import os, time, json, uuid, asyncio
from typing import List, Optional, Dict, Any, Deque, Tuple
from collections import deque, defaultdict

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# =====================
# Konfigurace z ENV
# =====================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
API_KEYS = set([k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()])
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))        # požadavků / min / klíč/IP
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "1"))                        # 1 = jednoduché, stabilní

APP_NAME = "jarvik-model-gateway"
APP_VERSION = "1.0.0"

# =====================
# Inicializace
# =====================
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting – jednoduché okno
_req_windows: Dict[str, Deque[float]] = defaultdict(deque)
WINDOW_SEC = 60.0

def _rate_key(api_key: str, ip: str) -> str:
    return f"{api_key or 'no-key'}|{ip}"

def _check_rate_limit(api_key: str, ip: str):
    now = time.time()
    key = _rate_key(api_key, ip)
    dq = _req_windows[key]
    # odmazat staré
    while dq and (now - dq[0]) > WINDOW_SEC:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    dq.append(now)

# Auth
def _auth_check(auth_header: Optional[str]) -> str:
    if not API_KEYS:
        # pokud nejsou nastaveny klíče, povolit pouze local (interní nasazení)
        return "no-auth"
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.split(None, 1)[1].strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token

# HTTP klient na Ollamu
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0), follow_redirects=True)

# =====================
# Pomocné mapování
# =====================
def _map_openai_to_ollama_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mapuje OpenAI /v1/chat/completions -> Ollama /api/chat
    """
    model = payload.get("model", "")
    messages = payload.get("messages", [])
    temperature = payload.get("temperature", None)
    top_p = payload.get("top_p", None)
    max_tokens = payload.get("max_tokens", None)
    stop = payload.get("stop", None)

    ollama_req = {
        "model": model,
        "messages": messages,
        "stream": payload.get("stream", False),
        "options": {}
    }
    # Mapování parametrů, pokud jsou
    if temperature is not None:
        ollama_req["options"]["temperature"] = float(temperature)
    if top_p is not None:
        ollama_req["options"]["top_p"] = float(top_p)
    if max_tokens is not None:
        ollama_req["options"]["num_predict"] = int(max_tokens)
    if stop is not None:
        ollama_req["options"]["stop"] = stop
    return ollama_req

def _openai_chunk(model: str, delta_content: Optional[str], finish_reason: Optional[str]) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": ({"role": "assistant"} if delta_content is None and finish_reason is None else {"content": delta_content}),
            "finish_reason": finish_reason
        }]
    }

def _openai_final_response(model: str, full_text: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop"
        }],
        "usage": {
            # Ollama nevrací přesně tokeny; lze doplnit heuristiku/odhad pokud chceš
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }

# =====================
# Health & Models
# =====================
@app.get("/healthz")
async def healthz():
    try:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        ok = r.status_code == 200
    except Exception:
        ok = False
    return {"name": APP_NAME, "version": APP_VERSION, "ollama_ok": ok}

@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None), request: Request = None):
    api_key = _auth_check(authorization)
    _check_rate_limit(api_key, request.client.host if request and request.client else "0.0.0.0")

    r = await client.get(f"{OLLAMA_URL}/api/tags")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Ollama not available")
    data = r.json().get("models", [])
    # OpenAI-compatible response
    return {
        "object": "list",
        "data": [{"id": m.get("name", ""), "object": "model", "created": None, "owned_by": "ollama"} for m in data]
    }

# =====================
# Chat Completions
# =====================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    api_key = _auth_check(authorization)
    _check_rate_limit(api_key, request.client.host if request and request.client else "0.0.0.0")

    payload = await request.json()
    ollama_payload = _map_openai_to_ollama_chat(payload)
    model = payload.get("model", "unknown")

    stream = bool(payload.get("stream", False))
    if not stream:
        # non-stream
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        data = r.json()
        msg = data.get("message", {})
        text = msg.get("content", "")
        return JSONResponse(_openai_final_response(model, text))

    # stream = True
    async def event_generator():
        # Přes Ollamu jako stream (server-sent chunks `data: {...}`)
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                raise HTTPException(status_code=502, detail=f"Ollama stream error: {detail.decode('utf-8','ignore')}")
            # první "role" delta (kompatibilita s některými klienty)
            yield f"data: {json.dumps(_openai_chunk(model, None, None))}\n\n"
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # Ollama posílá JSON po řádcích, někdy prefixed 'data: '
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("done"):
                    # finální prázdný delta s finish_reason=stop
                    yield f"data: {json.dumps(_openai_chunk(model, None, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                msg = obj.get("message", {})
                piece = msg.get("content", "")
                if piece:
                    yield f"data: {json.dumps(_openai_chunk(model, piece, None))}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =====================
# Embeddings
# =====================
@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(None)):
    api_key = _auth_check(authorization)
    _check_rate_limit(api_key, request.client.host if request and request.client else "0.0.0.0")

    payload = await request.json()
    model = payload.get("model", "")
    _input = payload.get("input")
    if _input is None:
        raise HTTPException(status_code=400, detail="Missing 'input'")

    texts: List[str]
    if isinstance(_input, list):
        texts = [str(t) for t in _input]
    else:
        texts = [str(_input)]

    results = []
    for t in texts:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": t})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {r.text}")
        emb = r.json().get("embedding", [])
        results.append({"object": "embedding", "embedding": emb, "index": len(results)})

    return {
        "object": "list",
        "data": results,
        "model": model,
        "usage": {"prompt_tokens": None, "total_tokens": None}
    }

# =====================
# Kořenové info
# =====================
@app.get("/")
async def root():
    return {"service": APP_NAME, "version": APP_VERSION, "endpoints": ["/healthz", "/v1/models", "/v1/chat/completions", "/v1/embeddings"]}

# =====================
# Spuštění lokálně
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8095, reload=False)
