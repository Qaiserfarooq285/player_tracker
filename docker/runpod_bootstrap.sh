#!/usr/bin/env bash
# PitchVision -- RunPod pod bootstrap (docs/DEPLOY.md).
#
# Set as the pod template's "Container Start Command":
#   bash -c "curl -fsSL https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh | bash"
#
# Everything that must survive a pod stop/restart lives on the network volume mounted at
# /workspace: the checkout, its .venv, models/ (1.9 GB), input/, work/ and output/. First boot
# takes ~10 min (deps + weights); every boot after that skips straight to starting the server.
#
# Environment (set on the RunPod template):
#   PV_ACCESS_PASSWORD       password for the web UI -- REQUIRED for anything public
#   CLOUDFLARE_TUNNEL_TOKEN  (optional) token from Cloudflare Zero Trust -> your own domain
#   GEMINI_API_KEY           (optional) same meaning as in .env.example
#   PORT                     listen port, default 8000 (expose the same port as HTTP on the pod)
#   PV_BRANCH / PV_REPO_URL  which code to run, default master of the public repo
#   PV_AUTO_UPDATE=0         keep whatever checkout is on the volume; default 1 = fast-forward
#                            to origin/$PV_BRANCH on each boot
set -euo pipefail

WS="${PV_WORKSPACE:-/workspace}"
APP="$WS/pitchvision"
REPO_URL="${PV_REPO_URL:-https://github.com/Qaiserfarooq285/payertracker.git}"
BRANCH="${PV_BRANCH:-master}"
export PORT="${PORT:-8000}"
PIP_EXTRAS='.[detect,team,ocr,jersey_parseq,api]'
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

log() { echo "[pitchvision] $*"; }

# Keep RunPod's own SSH / web-terminal services alive for debugging (the official images start
# them from /start.sh, which our start command replaces).
if [ -x /start.sh ] && [ "${PV_SKIP_RUNPOD_SERVICES:-0}" != "1" ]; then
  /start.sh >"$WS/runpod-start.log" 2>&1 &
fi

# --- system packages (ffmpeg for decode/encode; libGL for opencv) ---------------------------
need_apt=0
command -v ffmpeg >/dev/null 2>&1 || need_apt=1
command -v git >/dev/null 2>&1 || need_apt=1
ldconfig -p 2>/dev/null | grep -q 'libGL.so.1' || need_apt=1
if [ "$need_apt" = "1" ]; then
  log "installing system packages (ffmpeg, git, libGL)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ffmpeg git curl ca-certificates libgl1 libglib2.0-0 >/dev/null
fi

# --- code -----------------------------------------------------------------------------------
mkdir -p "$WS"
if [ ! -d "$APP/.git" ]; then
  log "cloning $REPO_URL ($BRANCH) -> $APP"
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
elif [ "${PV_AUTO_UPDATE:-1}" = "1" ]; then
  log "updating checkout to origin/$BRANCH"
  git -C "$APP" fetch --quiet origin "$BRANCH"
  git -C "$APP" reset --hard --quiet "origin/$BRANCH"
fi
cd "$APP"
log "running $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)"

# --- python env (on the volume, so it persists) ----------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
if [ ! -x .venv/bin/python ]; then
  log "creating .venv (python 3.11)"
  uv venv --quiet --python 3.11 .venv
fi
# Re-install only when pyproject.toml changes -- the stamp name carries its hash.
STAMP=".venv/.pitchvision-deps-$(sha256sum pyproject.toml | cut -c1-16)"
if [ ! -f "$STAMP" ]; then
  log "installing dependencies (first boot: several minutes)"
  uv pip install --quiet --python .venv/bin/python -e "$PIP_EXTRAS" --extra-index-url "$TORCH_INDEX"
  rm -f .venv/.pitchvision-deps-*
  touch "$STAMP"
fi

# --- model weights (verified by checksum; no-op once present) --------------------------------
.venv/bin/python scripts/download_models.py

# --- your domain, via Cloudflare Tunnel (optional) --------------------------------------------
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    log "installing cloudflared"
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
  fi
  cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" >"$WS/cloudflared.log" 2>&1 &
  log "cloudflared tunnel started (log: $WS/cloudflared.log)"
fi

if [ -z "${PV_ACCESS_PASSWORD:-}" ]; then
  log "WARNING: PV_ACCESS_PASSWORD is not set -- anyone with the URL can use this GPU"
fi

log "starting server on port $PORT"
exec .venv/bin/python start_app.py
