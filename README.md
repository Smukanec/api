# Jarvik Model Gateway

Jarvik Model Gateway je jednoduchá služba nad rámcem FastAPI, která poskytuje rozhraní kompatibilní s OpenAI pro server Ollama. Nabízí autentizaci API klíčem a omezení počtu požadavků.

## Vlastnosti
- Proxy pro endpointy serveru Ollama v OpenAI formátu.
- Bearer autentizace s konfigurovatelnými API klíči.
- Rate‑limiting dle dvojice API klíč/IP.
- CORS s konfigurovatelnými origins.
- Chat completions se SSE streamingem.
- Generování embeddingů.

## Instalace
1. Spusťte `install.sh` (vyžaduje root). Skript:
   - Nainstaluje závislosti a vytvoří virtuální prostředí.
   - Nakopíruje zdrojové soubory do `/opt/jarvik-model-gateway`.
   - Vytvoří `/etc/jarvik/model-gateway.env` s proměnnými prostředí.
   - Zaregistruje systemd službu naslouchající na `0.0.0.0:8095`.
   - Upraví UFW, aby port 8095 byl dostupný pouze z LAN.

2. Upravte `/etc/jarvik/model-gateway.env`:
   - `OLLAMA_URL`
   - `API_KEYS` (čárkami oddělené tokeny)
   - `ALLOWED_ORIGINS`
   - `RATE_LIMIT_PER_MIN`

3. Spusťte službu: `systemctl start jarvik-model-gateway`.

## Použití
Všechny endpointy vyžadují Bearer token z `API_KEYS` a jsou limitovány podle klíče/IP.

Endpointy:
- `GET /healthz` – kontrola dostupnosti služby a Ollamy.
- `GET /v1/models` – seznam modelů.
- `POST /v1/chat/completions` – generování chat odpovědí (SSE při `stream: true`).
- `POST /v1/embeddings` – výpočet embeddingů.
- `GET /` – základní informace o službě.

Příklad:
```bash
curl -H "Authorization: Bearer <token>" http://host:8095/v1/models
```

## Vývoj
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8095
```
