#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/jarvik-model-gateway"
USER_SVC="jarvik"
PY=python3

echo "===> Install system packages"
apt-get update -y
apt-get install -y ${PY} ${PY}-venv

echo "===> Create user and dirs"
id -u "${USER_SVC}" &>/dev/null || useradd -r -m -d /home/${USER_SVC} -s /usr/sbin/nologin ${USER_SVC}
mkdir -p "${APP_DIR}"
mkdir -p /etc/jarvik
chown -R ${USER_SVC}:${USER_SVC} "${APP_DIR}" /etc/jarvik

echo "===> Place app files"
# Očekává se, že app.py a requirements.txt jsou v aktuálním adresáři
cp app.py requirements.txt "${APP_DIR}/"
chown -R ${USER_SVC}:${USER_SVC} "${APP_DIR}"

echo "===> Python venv"
cd "${APP_DIR}"
${PY} -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "===> Environment file (skip if exists)"
ENV_FILE="/etc/jarvik/model-gateway.env"
if [ ! -f "$ENV_FILE" ]; then
  cat >/etc/jarvik/model-gateway.env <<'EOF'
OLLAMA_URL="http://127.0.0.1:11434"
API_KEYS="supersecret1"
ALLOWED_ORIGINS="*"
RATE_LIMIT_PER_MIN=120
EOF
  chown ${USER_SVC}:${USER_SVC} "$ENV_FILE"
  chmod 640 "$ENV_FILE"
fi

echo "===> Systemd unit"
cat >/etc/systemd/system/jarvik-model-gateway.service <<'EOF'
[Unit]
Description=Jarvik Model Gateway (OpenAI-compatible proxy to Ollama)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jarvik
Group=jarvik
WorkingDirectory=/opt/jarvik-model-gateway
EnvironmentFile=/etc/jarvik/model-gateway.env
ExecStart=/opt/jarvik-model-gateway/venv/bin/python app.py
Restart=always
RestartSec=2
RuntimeMaxSec=7d
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now jarvik-model-gateway

echo "===> Caddy snippet (manual step)"
echo "Přidej blok pro model.jarvik-ai.tech do /etc/caddy/Caddyfile a reloadni: sudo caddy fmt --overwrite /etc/caddy/Caddyfile && sudo systemctl reload caddy"

echo "===> Done. Test:"
echo "curl -s http://127.0.0.1:8095/healthz"
