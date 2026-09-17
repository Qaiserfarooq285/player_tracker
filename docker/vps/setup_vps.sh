#!/usr/bin/env bash
# PitchVision -- one-shot setup of the Hostinger KVM VPS as the HTTPS front door for the RunPod
# pod (docs/DEPLOY.md "VPS front door"). Idempotent: re-run it to point the domain at a new pod.
#
#   DOMAIN=app.example.com POD_ID=abc123xyz EMAIL=you@example.com bash setup_vps.sh
#
#   DOMAIN   the hostname whose A record points at this VPS
#   POD_ID   the RunPod pod id (the part before "-8000.proxy.runpod.net")
#   EMAIL    Let's Encrypt expiry notices (optional; --register-unsafely-without-email otherwise)
#   PORT     port the pod exposes as HTTP (default 8000)
#
# Runs as root on Ubuntu/Debian. Installs nginx + certbot, writes the vhost from
# pitchvision.nginx.conf, drops the "pod is stopped" page, and obtains/renews the certificate.
set -euo pipefail

: "${DOMAIN:?set DOMAIN=app.example.com}"
: "${POD_ID:?set POD_ID=<runpod pod id>}"
PORT="${PORT:-8000}"
UPSTREAM="${POD_ID}-${PORT}.proxy.runpod.net"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[pitchvision-vps] $*"; }

export DEBIAN_FRONTEND=noninteractive
if ! command -v nginx >/dev/null 2>&1 || ! command -v certbot >/dev/null 2>&1; then
  log "installing nginx + certbot"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends nginx certbot python3-certbot-nginx >/dev/null
fi

log "writing vhost for $DOMAIN -> https://$UPSTREAM"
mkdir -p /var/www/pitchvision
install -m 0644 "$HERE/pod-offline.html" /var/www/pitchvision/pod-offline.html
sed -e "s|__DOMAIN__|$DOMAIN|g" -e "s|__UPSTREAM__|$UPSTREAM|g" \
  "$HERE/pitchvision.nginx.conf" >/etc/nginx/sites-available/pitchvision
ln -sf /etc/nginx/sites-available/pitchvision /etc/nginx/sites-enabled/pitchvision
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx >/dev/null
systemctl reload nginx

# Certificate: only once the A record resolves here, otherwise certbot's HTTP challenge fails.
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  log "requesting Let's Encrypt certificate for $DOMAIN"
  if [ -n "${EMAIL:-}" ]; then
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
  else
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect
  fi
else
  # Re-run after a POD_ID change: the vhost was rewritten without the TLS block certbot added,
  # so re-apply it (certbot is a no-op on the cert itself when it's still valid).
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --reinstall --redirect >/dev/null
  systemctl reload nginx
fi

if [ -f /etc/ufw/ufw.conf ] && ufw status | grep -q "Status: active"; then
  ufw allow 'Nginx Full' >/dev/null || true
fi

log "done: https://$DOMAIN -> $UPSTREAM"
log "health check: curl -s https://$DOMAIN/api/health"
