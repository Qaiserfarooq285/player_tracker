# Hosting PitchVision on RunPod with your own domain

This is the step-by-step for the owner's setup: **RunPod credit** for the GPU and a **Hostinger
domain** for the address. Nothing here needs a Docker image build or a server you have to
maintain — a RunPod pod pulls the code straight from GitHub on every boot.

```
  browser ──https──▶ Cloudflare (your domain, free)
                        │  Cloudflare Tunnel (outbound from the pod; no open ports, no pod IP)
                        ▼
                   RunPod GPU pod  ── runs docker/runpod_bootstrap.sh
                        │             └─ FastAPI + web UI on :8000  (start_app.py, no --reload)
                        ▼
                   /workspace  (RunPod NETWORK VOLUME — survives stop/restart)
                     └─ pitchvision/   checkout · .venv · models/ · input/ · work/ · output/
```

Why this shape:
- **Pod + network volume, not serverless.** A run takes minutes and the UI polls a long-lived
  server; serverless workers are for short stateless calls. The volume is what lets you **stop
  the pod to stop paying** without losing uploads, outputs or the 1.9 GB of model weights.
- **Cloudflare Tunnel, not DNS → pod IP.** A pod's IP and ports change on every restart and it
  has no port 443; the tunnel gives a stable `https://app.yourdomain.com` for free.
- **Password gate in the app** (`PV_ACCESS_PASSWORD`). The RunPod proxy URL is public; without
  the gate anyone who finds it can upload videos and burn your credit.
- **Chunked uploads.** Cloudflare rejects any single request over 100 MB; the UI now uploads in
  24 MB slices, so 4K clips work through the domain.

---

## 0. Before you start (5 min)

| You need | Where |
|---|---|
| The repo on GitHub, public | already: `github.com/Qaiserfarooq285/payertracker` (`master`) |
| RunPod account with credit | runpod.io |
| Cloudflare account (free) | dash.cloudflare.com — sign up |
| Access to the Hostinger domain's DNS settings | hpanel.hostinger.com |
| A long access password you'll set on the pod | make one up now; you'll type it into the UI |

Optional: `GEMINI_API_KEY` (only the celebration / key-moment stage uses it).

---

## 1. RunPod — network volume (2 min)

1. RunPod → **Storage** → **New Network Volume**.
2. Pick a **region that lists the GPU you want** (volumes are region-locked; check *Pods →
   Deploy* first to see which region has, e.g., RTX 4090 availability).
3. Size **60 GB** (models 2 GB + Python env 7 GB + your videos and outputs). You can grow it
   later; you cannot shrink it. Cost is per GB-month, roughly the price of a coffee.
4. Name it `pitchvision-data`.

## 2. RunPod — pod template (5 min)

RunPod → **Templates** → **New Template**:

| Field | Value |
|---|---|
| Template type | Pod |
| Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (any `runpod/pytorch` CUDA ≥ 12.4 image works — the bootstrap builds its own Python env) |
| Container disk | 20 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` (optional, for SSH debugging) |
| **Container start command** | `bash -c "curl -fsSL https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh \| bash"` — copy it from the top of [`docker/runpod_bootstrap.sh`](../docker/runpod_bootstrap.sh) without the `\|` escaping |
| Environment variables | see [`docker/runpod.env.example`](../docker/runpod.env.example): **`PV_ACCESS_PASSWORD`** (optional — defaults to `admin1122` if left unset, change it), `RUNPOD_API_KEY` (optional, enables idle auto-stop — §"Idle auto-stop" below), `CLOUDFLARE_TUNNEL_TOKEN` (after step 4), `GEMINI_API_KEY` (optional) |

Save the template.

## 3. RunPod — deploy the pod (first boot ≈ 10 min)

1. **Pods → Deploy**. GPU: the pipeline was built for a 12 GB card, so anything from
   **RTX A4000 (16 GB)** up works; **RTX 3090 / 4090 (24 GB)** are the sweet spot for speed
   per dollar. Prefer *Secure Cloud* for reliability; *Community Cloud* is cheaper.
2. **Select your network volume** (`pitchvision-data`) — this is the step people miss; without
   it everything is wiped on restart.
3. Choose the template from step 2 → **Deploy**.
4. Open the pod's **Logs**. You'll see `[pitchvision] cloning…`, `installing dependencies
   (first boot: several minutes)`, `downloading …`, then `starting server on port 8000`.
5. **Connect → HTTP Service [Port 8000]** opens `https://<POD_ID>-8000.proxy.runpod.net`.
   Enter your access password. Upload a clip, run it — this is the full app on the GPU.

Every later boot skips the install and starts in ~30 s.

### Login

The web UI is gated by a single shared password — the whole point is "after deployment it
should be used by us only", not a per-user account system. **The gate is ON out of the box**:
leave `PV_ACCESS_PASSWORD` unset and the pod accepts the built-in default, **`admin1122`**. Set
`PV_ACCESS_PASSWORD` on the template to your own password to change it (Restart to apply).
`PV_ACCESS_PASSWORD=off` disables the gate entirely — only ever do that for local editing on
your own machine, never on a hosted pod.

## 4. Your Hostinger domain → the pod (Cloudflare Tunnel, 15 min + DNS wait)

### 4a. Put the domain on Cloudflare
1. Cloudflare dashboard → **Add a site** → type your domain → **Free** plan → Continue.
   Cloudflare scans and imports existing DNS records; Continue.
2. It shows **two nameservers** (like `ada.ns.cloudflare.com` / `rob.ns.cloudflare.com`).
3. Hostinger hPanel → **Domains** → your domain → **DNS / Nameservers** → **Change
   nameservers** → *Custom* → paste the two Cloudflare nameservers → Save.
4. Wait for Cloudflare's "site is active" email (usually under an hour, up to 24 h).
   Email/website you already host at Hostinger keeps working — the imported records carry over.

### 4b. Create the tunnel
1. **one.dash.cloudflare.com** (Zero Trust) → **Networks → Tunnels → Create a tunnel** →
   *Cloudflared* → name `pitchvision` → Save.
2. On the install page, ignore the OS buttons; in the command shown, copy the long string after
   `--token`. That is your `CLOUDFLARE_TUNNEL_TOKEN`.
3. **Public hostname** tab → Add: subdomain `app`, domain yours, type **HTTP**, URL
   `localhost:8000` → Save.
4. RunPod → your template (or the running pod's *Edit*) → add env var
   `CLOUDFLARE_TUNNEL_TOKEN=<the token>` → **Restart** the pod.
5. Open **https://app.yourdomain.com**. Cloudflare provides the HTTPS certificate automatically.

### 4c. Optional extra lock (recommended if others will use it)
Zero Trust → **Access → Applications → Add** → Self-hosted → domain `app.yourdomain.com` →
policy *Allow* → include *Emails* = the addresses you permit. Visitors then get a one-time email
code before they even reach the app's own password. Free up to 50 users. (The RunPod proxy URL
bypasses Cloudflare, which is why the in-app password still matters.)

---

## Day-to-day

| Task | How |
|---|---|
| **Stop paying** when not in use | Pod → **Stop**. Volume cost only. **Start** brings it back with all data (~30 s). |
| Update the app | `git push` to `master` → Pod → **Restart** (the bootstrap fast-forwards the checkout). Pin a branch with `PV_BRANCH`; freeze with `PV_AUTO_UPDATE=0`. |
| Change the password | edit `PV_ACCESS_PASSWORD` on the pod → Restart. Sessions are invalidated on every restart anyway. |
| Free disk | SSH / web terminal: `rm -rf /workspace/pitchvision/work/*` (cached intermediates; re-created on demand). Delete old `output/<slug>/` folders you no longer need. |
| Watch logs | Pod → Logs; tunnel log at `/workspace/cloudflared.log`; RunPod's own services at `/workspace/runpod-start.log`. |
| Health | `https://app.yourdomain.com/api/health` (public, no login) shows the GPU the server sees. |

### Idle auto-stop (recommended — saves credit)

Forgetting to press Stop is the single most expensive mistake with a metered pod. Set it up once
and the pod stops itself:

1. RunPod → **Settings → API Keys → Create API Key** → type *Restricted* → give it the **pods**
   permission, read/write → Create → copy the key.
2. Paste it into the template's `RUNPOD_API_KEY` env var → Restart the pod. `RUNPOD_POD_ID` needs
   no setup — RunPod injects it into every pod automatically.
3. That's it. The pod now stops itself after `PV_IDLE_STOP_MINUTES` (default 30) with **no HTTP
   requests and no pipeline job running** — checked every minute in the background
   (`apps/api/idle_stop.py`). Nothing is lost: the network volume (checkout, `input/`, `work/`,
   `output/`) persists across a stop exactly like a manual Stop. **Start** it again from the
   RunPod console when you're back.
4. `https://app.yourdomain.com/api/health` → `idle_stop.enabled` confirms it's armed (`false`
   until `RUNPOD_API_KEY` is filled in — off by default, matching the blank env template).

## Troubleshooting

- **Login overlay never accepts the password** → the pod's `PV_ACCESS_PASSWORD` has trailing
  spaces or you edited it without restarting. Check the pod log's startup banner: `Access gate: ON`
  (or `ON (default password...)` if you left it unset — the gate is on either way).
- **`/api/health` says `gpu_available: false`** → the pod was deployed without a GPU or the
  driver is older than the CUDA 12.4 wheels; redeploy on a different host.
- **Upload dies at ~100 MB** → you're hitting Cloudflare's per-request cap with the old
  single-request path. The current UI uses `/api/upload/chunk`; hard-refresh the page
  (Ctrl+Shift+R) to drop the cached old `app.js`.
- **`Job failed` right after a restart** → any run in flight dies when the pod restarts. Don't
  restart mid-run; results already written to `output/` are kept.
- **First boot stuck on `installing dependencies`** → normal for 5–10 min. If it exceeds 20 min,
  check the log for a pip error and restart; the install resumes from uv's cache.
- **Tunnel shows "inactive"** → the token env var is missing/wrong on the pod, or the pod is
  stopped. `cat /workspace/cloudflared.log`.
- **`409 another run is already in flight`** → the same video is being processed already; wait
  for it or process a different upload (one GPU, one job at a time by design).

## Costs, honestly (check current RunPod prices — they move)

- GPU pod: roughly **$0.20–0.70 / hour while running**, by card. Stop it when idle, or set up
  **idle auto-stop** above so a forgotten tab doesn't run the meter overnight.
- Network volume: **~$0.07 / GB-month** → 60 GB ≈ $4 / month.
- Cloudflare (domain proxy, tunnel, Access ≤ 50 users): **$0**.
- Hostinger: whatever the domain renewal already costs; no hosting plan is needed for this.

## Running it on your own PC instead (unchanged)

`make serve` (or `.venv/bin/python start_app.py`) — no password, no tunnel, `http://localhost:8000`.
Set `PV_DEV=1` only while editing code (auto-reload); it kills in-flight jobs otherwise.
