#!/usr/bin/env python3
import os, time, json, uuid
from typing import Optional, Dict, Any, List, Literal, Union, Tuple
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ===== Konfigurace =====
class Settings(BaseSettings):
    ollama_url: str = "http://127.0.0.1:11434"
    api_keys: str = ""
    allowed_origins: str = "*"
    rate_limit_per_min: int = 120
    cache_ttl: int = 60

settings = Settings()

OLLAMA_URL = settings.ollama_url
API_KEYS = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
ALLOWED_ORIGINS = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
RATE_LIMIT_PER_MIN = settings.rate_limit_per_min
MODELS_TTL = settings.cache_ttl

APP_NAME = "jarvik-model-gateway"
APP_VERSION = "1.0.0-lan"

# ===== App & CORS =====
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        app.state.client = client
        yield

app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
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
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEYS empty")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.split(None, 1)[1].strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token

async def verify_request(
    request: Request,
    authorization: str = Header(None)
) -> str:
    api_key = _auth_check(authorization)
    _check_rl(api_key, request.client.host if request and request.client else "")
    return api_key

# ===== Mapování OpenAI↔Ollama =====
def _map_openai_to_ollama_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = payload.get("model", "")
    messages = payload.get("messages", [])
    req = {"model": model, "messages": messages, "stream": bool(payload.get("stream", False)), "options": {}}
    if (t := payload.get("temperature")) is not None: req["options"]["temperature"] = float(t)
    if (p := payload.get("top_p")) is not None: req["options"]["top_p"] = float(p)
    if (mt := payload.get("max_tokens")) is not None: req["options"]["num_predict"] = int(mt)
    if (stop := payload.get("stop")) is not None: req["options"]["stop"] = stop
    return req

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

# ===== Pydantic modely =====
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: List[Choice]
    usage: Optional[Dict[str, Any]] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, Any]

class ModelItem(BaseModel):
    id: str
    object: Literal["model"] = "model"

class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelItem]

# ===== Cache modelů =====
_models_cache: Dict[str, Tuple[float, Any]] = {}
MODELS_CACHE_KEY = "ollama/models"

async def get_models_cached(request: Request):
    now = time.time()
    cached = _models_cache.get(MODELS_CACHE_KEY)
    if cached and now - cached[0] < MODELS_TTL:
        return cached[1]
    client: httpx.AsyncClient = request.app.state.client
    r = await client.get(f"{OLLAMA_URL}/api/tags")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Ollama not available")
    data = r.json().get("models", [])
    _models_cache[MODELS_CACHE_KEY] = (now, data)
    return data

# ===== Health & modely =====
@app.get("/healthz")
async def healthz(request: Request):
    client: httpx.AsyncClient = request.app.state.client
    try:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        ok = r.status_code == 200
    except Exception:
        ok = False
    return {"name": APP_NAME, "version": APP_VERSION, "ollama_ok": ok}

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request, api_key: str = Depends(verify_request)):
    data = await get_models_cached(request)
    return ModelListResponse(
        object="list",
        data=[ModelItem(id=m.get("name", "")) for m in data]
    )

# ===== Chat completions =====
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest, request: Request, api_key: str = Depends(verify_request)):
    client: httpx.AsyncClient = request.app.state.client
    model = req.model
    ollama_payload = _map_openai_to_ollama_chat(req.dict())

    if not req.stream:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        text = r.json().get("message", {}).get("content", "")
        return ChatResponse(**_openai_final(model, text))

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
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest, request: Request, api_key: str = Depends(verify_request)):
    client: httpx.AsyncClient = request.app.state.client
    texts: List[str] = [str(t) for t in (req.input if isinstance(req.input, list) else [req.input])]
    results = []
    for t in texts:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": req.model, "prompt": t})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {r.text}")
        emb = r.json().get("embedding", [])
        results.append(EmbeddingData(embedding=emb, index=len(results)))
    return EmbeddingResponse(
        object="list",
        data=results,
        model=req.model,
        usage={"prompt_tokens": None, "total_tokens": None}
    )

@app.get("/")
async def root():
    return {"service": APP_NAME, "version": APP_VERSION, "mode": "LAN", "endpoints": ["/healthz", "/v1/models", "/v1/chat/completions", "/v1/embeddings"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8095, reload=False)
