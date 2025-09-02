# Jarvik Model Gateway

This repository provides a small FastAPI application that exposes an OpenAI-compatible API proxying requests to an [Ollama](https://ollama.com/) server.

## Installation

Clone the repository and install dependencies inside a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For a production setup with systemd and firewall rules, review the provided `install.sh` script.

## Running

Start the API locally:

```bash
python app.py
```

The service listens on port `8095` by default.

## Endpoints

- `GET /healthz` – basic health check
- `GET /v1/models` – list available models
- `POST /v1/chat/completions` – create chat completions (supports streaming)
- `POST /v1/embeddings` – generate embeddings

All endpoints require a Bearer token defined in the `API_KEYS` environment variable.

## Running Local Models via Docker

1. Verify locally available images:

```bash
docker images
```

Expected list:
- at-code:latest
- chat-qwen:latest
- smollm:1.7b
- codegemma:2b
- qwen3:1.7b
- codegemma2b-tuned:latest
- stable-code:3b

2. Start a container for a model:

```bash
docker run -it --rm --gpus all -p HOST_PORT:8000 IMAGE_NAME
```

### Example commands

```bash
docker run -it --rm --gpus all -p 8000:8000 at-code:latest
docker run -it --rm --gpus all -p 8001:8000 chat-qwen:latest
docker run -it --rm --gpus all -p 8002:8000 smollm:1.7b
docker run -it --rm --gpus all -p 8003:8000 codegemma:2b
docker run -it --rm --gpus all -p 8004:8000 qwen3:1.7b
docker run -it --rm --gpus all -p 8005:8000 codegemma2b-tuned:latest
docker run -it --rm --gpus all -p 8006:8000 stable-code:3b
```

3. Test a running model:

```bash
curl http://localhost:HOST_PORT
```

### Additional tips
- `--gpus all` requires Docker with GPU support.
- Adjust host ports (`-p HOST_PORT:8000`) to avoid conflicts.
- Use `docker logs <container_id>` for debugging.
- Restrict public access with firewall rules or authentication when exposing ports.

