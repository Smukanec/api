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
