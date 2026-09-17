#!/usr/bin/env bash
# PitchVision -- push the current master to the VPS gateway (docs/DEPLOY.md "Always-on gateway").
# The pod side needs nothing: it fast-forwards to origin/master on its next boot, and the gateway
# recreates it whenever it can't be started. Run from the owner's Mac after `git push`:
#
#   bash docker/deploy_vps.sh
#
# Reads (all already on the owner's machine after the first deployment):
#   ~/.ssh/pitchvision_github_token.txt   GITHUB_TOKEN (private repo, read-only)
#   ~/.ssh/pitchvision_site_password.txt  PV_ACCESS_PASSWORD
#   ~/.ssh/pitchvision_runpod_key.txt     RUNPOD_API_KEY (or the env var)
#   ~/.ssh/pitchvision_vps(.pub)          SSH key for the VPS; the .pub is installed on the pod too
set -euo pipefail

DOMAIN="${DOMAIN:-app.thereachvision.tech}"
VPS="${VPS:-root@187.6.165.147}"
VPS_KEY="${VPS_KEY:-$HOME/.ssh/pitchvision_vps}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "$HOME/.ssh/pitchvision_runpod_key.txt")}"
GITHUB_TOKEN="${GITHUB_TOKEN:-$(cat "$HOME/.ssh/pitchvision_github_token.txt")}"
PASSWORD="$(cat "$HOME/.ssh/pitchvision_site_password.txt")"
SSH_PUB="$(cat "$VPS_KEY.pub")"

log() { echo "[deploy-vps] $*"; }
ssh_vps() { ssh -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new "$VPS" "$@"; }

# The setup script lives in the repo; the VPS keeps its own checkout, but the FIRST run needs the
# script before any checkout exists -- so always ship the current copy over first.
log "copying setup script to the VPS"
ssh_vps "mkdir -p /root/pitchvision-vps"
scp -q -i "$VPS_KEY" "$(dirname "${BASH_SOURCE[0]}")/vps/setup_vps.sh" "$VPS:/root/pitchvision-vps/setup_vps.sh"

log "running setup on the VPS ($DOMAIN)"
ssh_vps "DOMAIN=$(printf %q "$DOMAIN") GITHUB_TOKEN=$(printf %q "$GITHUB_TOKEN") RUNPOD_API_KEY=$(printf %q "$RUNPOD_API_KEY") \
  PV_ACCESS_PASSWORD=$(printf %q "$PASSWORD") PV_POD_SSH_PUBLIC_KEY=$(printf %q "$SSH_PUB") \
  ${GEMINI_API_KEY:+GEMINI_API_KEY=$(printf %q "$GEMINI_API_KEY")} \
  bash /root/pitchvision-vps/setup_vps.sh"

log "health: $(curl -s -m 15 "https://$DOMAIN/api/health")"
