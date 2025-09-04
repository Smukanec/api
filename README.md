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
- `GET /v1/models` – list available models including `name`, `id`, `size` and `modified`
- `POST /v1/chat/completions` – create chat completions (supports streaming)
- `POST /v1/embeddings` – generate embeddings

All endpoints require a Bearer token defined in the `API_KEYS` environment variable.

## Running Local Models via Docker

1. Verify locally available images:

```bash
docker images
```

Expected list:
- everythinglm:13b-16k
- jarvik-chat:latest
- jarvik-coder:latest
- jarvik-rag:latest
- CognitiveComputations/dolphin-llama3.1:8b
- dolphincoder:15b-starcoder2-q5_K_M
- gpt-oss:latest
- starcoder:7b
- mistral:7b
- codellama:7b-instruct
- nous-hermes2:latest
- command-r:latest
- llama3:8b

2. Start a container for a model:

```bash
docker run -it --rm --gpus all -p HOST_PORT:8000 IMAGE_NAME
```

### Example commands

```bash
docker run -it --rm --gpus all -p 8000:8000 everythinglm:13b-16k
docker run -it --rm --gpus all -p 8001:8000 jarvik-chat:latest
docker run -it --rm --gpus all -p 8002:8000 jarvik-coder:latest
docker run -it --rm --gpus all -p 8003:8000 jarvik-rag:latest
docker run -it --rm --gpus all -p 8004:8000 CognitiveComputations/dolphin-llama3.1:8b
docker run -it --rm --gpus all -p 8005:8000 dolphincoder:15b-starcoder2-q5_K_M
docker run -it --rm --gpus all -p 8006:8000 gpt-oss:latest
docker run -it --rm --gpus all -p 8007:8000 starcoder:7b
docker run -it --rm --gpus all -p 8008:8000 mistral:7b
docker run -it --rm --gpus all -p 8009:8000 codellama:7b-instruct
docker run -it --rm --gpus all -p 8010:8000 nous-hermes2:latest
docker run -it --rm --gpus all -p 8011:8000 command-r:latest
docker run -it --rm --gpus all -p 8012:8000 llama3:8b
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

