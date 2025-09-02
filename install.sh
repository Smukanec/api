#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/jarvik-model-gateway"
PY=python3
LAN_CIDR="${LAN_CIDR:-10.0.1.0/24}"   # uprav dle své LAN

echo "===> Packages"
apt-get update -y
apt-get install -y ${PY} ${PY}-venv ufw

echo "===> App dir"
mkdir -p "$APP_DIR"
cp app.py requirements.txt "$APP_DIR"
cd "$APP_DIR"

echo "===> Venv"
${PY} -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "===> Env (skip pokud existuje)"
if [ ! -f /etc/jarvik/model-gateway.env ]; then
  mkdir -p /etc/jarvik
  cat >/etc/jarvik/model-gateway.env <<'EOF'
OLLAMA_URL="http://127.0.0.1:11434"
API_KEYS="mojelokalnikurvitko"
ALLOWED_ORIGINS="*"
RATE_LIMIT_PER_MIN=120
EOF
  chmod 640 /etc/jarvik/model-gateway.env
fi

echo "===> systemd unit"
cat >/etc/systemd/system/jarvik-model-gateway.service <<'EOF'
[Unit]
Description=Jarvik Model Gateway (OpenAI-compatible proxy to Ollama) - LAN
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

echo "===> UFW – povol jen LAN na 8095, ostatní deny"
ufw allow from "${LAN_CIDR}" to any port 8095 proto tcp
ufw deny 8095/tcp || true
ufw reload || true

echo "===> Done"
echo "Test: curl -s http://<LAN-IP>:8095/healthz"
