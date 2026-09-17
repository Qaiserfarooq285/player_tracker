#!/usr/bin/env bash
# PitchVision -- bring the site back when the pod can't Start (docs/DEPLOY.md "Pod won't start").
#
# A STOPPED RunPod pod does not reserve its GPU: after an idle auto-stop, Start can fail with
# "not enough free GPUs on the host machine". This terminates the dead pod, creates a fresh one on
# whatever host has a GPU (the network volume -- code, venv, models, uploads, outputs -- carries
# over), and points the VPS at the new pod id. Run it from the owner's Mac:
#
#   bash docker/redeploy.sh
#
# Reads (all already on the owner's machine after the first deployment):
#   ~/.ssh/pitchvision_github_token.txt   GITHUB_TOKEN (private repo, read-only)
#   ~/.ssh/pitchvision_site_password.txt  PV_ACCESS_PASSWORD
#   ~/.ssh/pitchvision_vps                SSH key for the VPS (and the pod)
#   RUNPOD_API_KEY                        env var, or ~/.ssh/pitchvision_runpod_key.txt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN="${DOMAIN:-app.thereachvision.tech}"
VPS="${VPS:-root@187.6.165.147}"
VPS_KEY="${VPS_KEY:-$HOME/.ssh/pitchvision_vps}"
GPUS="${GPUS:-NVIDIA L4,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA GeForce RTX 4090}"
DATACENTER="${DATACENTER:-EUR-IS-1}"
PY="${PY:-/usr/bin/python3}"   # the python.org build on the Mac lacks CA certs; the system one works

export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "$HOME/.ssh/pitchvision_runpod_key.txt")}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$(cat "$HOME/.ssh/pitchvision_github_token.txt")}"
PASSWORD="$(cat "$HOME/.ssh/pitchvision_site_password.txt")"

log() { echo "[redeploy] $*"; }
api() { curl -s -A pitchvision-redeploy -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }

# 1. If a pod exists and is merely stopped, one Start attempt is the cheap path (same pod id).
existing="$(api https://rest.runpod.io/v1/pods | "$PY" -c 'import sys,json; ps=[p for p in json.load(sys.stdin) if p.get("name")=="pitchvision"]; print(ps[0]["id"] if ps else "")')"
if [ -n "$existing" ]; then
  if api -X POST -d '{}' "https://rest.runpod.io/v1/pods/$existing/start" | grep -q '"desiredStatus":"RUNNING"'; then
    log "pod $existing started in place"
    POD_ID="$existing"
  else
    log "pod $existing cannot start (GPU taken) -- terminating it; the volume is kept"
    for _ in 1 2 3; do
      api -X DELETE "https://rest.runpod.io/v1/pods/$existing" >/dev/null || true
      sleep 6
      api https://rest.runpod.io/v1/pods | grep -q "\"$existing\"" || break
    done
  fi
fi

# 2. Fresh pod on any host with a GPU from the preference list.
if [ -z "${POD_ID:-}" ]; then
  POD_ID="$("$PY" "$HERE/runpod_provision.py" --password "$PASSWORD" --gpu "$GPUS" --datacenter "$DATACENTER" \
      --cloud SECURE --ssh-public-key "$(cat "$VPS_KEY.pub")" | sed -n 's/^POD_ID=//p')"
  [ -n "$POD_ID" ] || { log "provisioning failed"; exit 1; }
fi

# 3. Point the domain at it.
log "pointing $DOMAIN at pod $POD_ID"
ssh -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new "$VPS" \
  "DOMAIN=$DOMAIN POD_ID=$POD_ID bash /root/pitchvision-vps/setup_vps.sh 2>&1 | tail -1"

# 4. Wait for the app (fast boot: deps + models are on the volume).
for _ in $(seq 1 40); do
  if [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/health")" = "200" ]; then
    log "https://$DOMAIN is up on pod $POD_ID"; exit 0
  fi
  sleep 15
done
log "still not healthy after 10 min -- check the pod logs in the RunPod console"; exit 1
