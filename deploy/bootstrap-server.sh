#!/usr/bin/env bash
set -euo pipefail

app_root=/opt/digital-person
state_root=/var/lib/digital-mirror
environment_file=/etc/digital-mirror.env

if [[ ! -f "${app_root}/pyproject.toml" ]]; then
  echo "Expected application source at ${app_root}" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv nginx certbot python3-certbot-nginx

if ! id digital-mirror >/dev/null 2>&1; then
  useradd --system --home-dir "${state_root}" --create-home --shell /usr/sbin/nologin digital-mirror
fi

install -d -o digital-mirror -g digital-mirror -m 0750 "${state_root}/episodes"
python3 -m venv "${app_root}/.venv"
"${app_root}/.venv/bin/python" -m pip install --upgrade pip
"${app_root}/.venv/bin/python" -m pip install "${app_root}"

if [[ ! -f "${environment_file}" ]]; then
  access_password="$(openssl rand -base64 24 | tr -d '\n')"
  session_secret="$(openssl rand -hex 48)"
  install -m 0600 /dev/null "${environment_file}"
  {
    echo "DIGITAL_MIRROR_ACCESS_PASSWORD=${access_password}"
    echo "DIGITAL_MIRROR_SESSION_SECRET=${session_secret}"
    echo "DIGITAL_MIRROR_COOKIE_SECURE=true"
    echo "DIGITAL_MIRROR_EPISODE_ROOT=${state_root}/episodes"
  } >> "${environment_file}"
fi

install -m 0644 "${app_root}/deploy/digital-mirror.service" /etc/systemd/system/digital-mirror.service
install -m 0644 "${app_root}/deploy/nginx-mirror.conf" /etc/nginx/sites-available/mirror.learnpath.tech
ln -sfn /etc/nginx/sites-available/mirror.learnpath.tech /etc/nginx/sites-enabled/mirror.learnpath.tech

systemctl daemon-reload
systemctl enable --now digital-mirror nginx
nginx -t
systemctl reload nginx

curl --fail --silent --show-error http://127.0.0.1:8010/api/health
