#!/usr/bin/env python3
import os, time, json, uuid
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ===== Konfigurace z ENV =====
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

APP_NAME = "jarvik-model-gateway"
APP_VERSION = "1.0.0-local"

# ===== App & CORS =====
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Rate limit (klíč/IP) =====
WINDOW_SEC = 60.0
_req_windows: Dict[str, deque] = defaultdict(deque)

def _rk(api_key: str, ip: str) -> str:
    return f"{api_key or 'no-key'}|{ip or '0.0.0.0'}"

def _check_rl(api_key: str, ip: str):
    now = time.time()
    dq = _req_windows[_rk(api_key, ip)]
    while dq and now - dq[0] > WINDOW_SEC:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    dq.append(now)

# ===== Auth „kurvítko“ =====
def _auth_check(auth_header: Optional[str]) -> str:
    if not API_KEYS:
        # Lokální režim vyžaduje klíč – bez něj fail
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEYS empty")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.split(None, 1)[1].strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token

# ===== HTTP klient na Ollamu =====
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0), follow_redirects=True)

# ===== Mapování OpenAI↔Ollama =====
def _map_openai_to_ollama_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = payload.get("model", "")
    messages = payload.get("messages", [])
    ollama_req = {
        "model": model,
        "messages": messages,
        "stream": payload.get("stream", False),
        "options": {}
    }
    # volitelné parametry
    if (t := payload.get("temperature")) is not None:
        ollama_req["options"]["temperature"] = float(t)
    if (p := payload.get("top_p")) is not None:
        ollama_req["options"]["top_p"] = float(p)
    if (mt := payload.get("max_tokens")) is not None:
        ollama_req["options"]["num_predict"] = int(mt)
    if (stop := payload.get("stop")) is not None:
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

def _openai_final(model: str, text: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    }

# ===== Health & modely =====
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
    _check_rl(api_key, request.client.host if request and request.client else "")
    r = await client.get(f"{OLLAMA_URL}/api/tags")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Ollama not available")
    data = r.json().get("models", [])
    return {"object": "list", "data": [{"id": m.get("name",""), "object": "model"} for m in data]}

# ===== Chat completions =====
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    api_key = _auth_check(authorization)
    _check_rl(api_key, request.client.host if request and request.client else "")
    payload = await request.json()
    model = payload.get("model", "unknown")
    ollama_payload = _map_openai_to_ollama_chat(payload)

    if not payload.get("stream", False):
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        text = r.json().get("message", {}).get("content", "")
        return JSONResponse(_openai_final(model, text))

    async def event_gen():
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                raise HTTPException(status_code=502, detail=f"Ollama stream error: {detail.decode('utf-8','ignore')}")
            yield f"data: {json.dumps(_openai_chunk(model, None, None))}\n\n"
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("done"):
                    yield f"data: {json.dumps(_openai_chunk(model, None, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield f"data: {json.dumps(_openai_chunk(model, piece, None))}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ===== Embeddings =====
@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(None)):
    api_key = _auth_check(authorization)
    _check_rl(api_key, request.client.host if request and request.client else "")
    payload = await request.json()
    model = payload.get("model", "")
    _input = payload.get("input")
    if _input is None:
        raise HTTPException(status_code=400, detail="Missing 'input'")
    texts: List[str] = [str(t) for t in (_input if isinstance(_input, list) else [_input])]
    results = []
    for t in texts:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": t})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {r.text}")
        emb = r.json().get("embedding", [])
        results.append({"object": "embedding", "embedding": emb, "index": len(results)})
    return {"object": "list", "data": results, "model": model, "usage": {"prompt_tokens": None, "total_tokens": None}}

@app.get("/")
async def root():
    return {"service": APP_NAME, "version": APP_VERSION, "mode": "LOCAL_ONLY", "endpoints": ["/healthz", "/v1/models", "/v1/chat/completions", "/v1/embeddings"]}

if __name__ == "__main__":
    # *** DŮLEŽITÉ: Běží POUZE na 127.0.0.1 ***
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8095, reload=False)
