#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/jarvik-model-gateway"
PY=python3

apt-get update -y
apt-get install -y ${PY} ${PY}-venv

mkdir -p "$APP_DIR"
cp app.py requirements.txt "$APP_DIR"
cd "$APP_DIR"

${PY} -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# env se připraví zvlášť (viz výš)

cat >/etc/systemd/system/jarvik-model-gateway.service <<'EOF'
[Unit]
Description=Jarvik Model Gateway (OpenAI-compatible proxy to Ollama) - LOCAL ONLY
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=root
Group=root
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

echo "== Test =="
echo "curl -s http://127.0.0.1:8095/healthz"
