#!/usr/bin/env python3
"""Unified Launcher for AI Football Player Tracking & Analytics Web Server.

Environment knobs (all optional; docs/DEPLOY.md):
  PORT     -- listen port (default 8000; RunPod's HTTP proxy expects the port you expose there)
  PV_DEV=1 -- enable uvicorn's auto-reload. OFF by default on purpose: the reloader restarts the
              server whenever a file under the repo changes, which kills any pipeline job that is
              mid-run (observed 2026-09-15 as a bare "Job failed: None"). Turn it on only while
              editing code locally, never on a hosted pod.
"""

import os
import sys
from pathlib import Path

import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("PV_DEV", "") == "1"
    gated = bool(os.environ.get("PV_ACCESS_PASSWORD", "").strip())
    print("=" * 70)
    print("  PITCHVISION AI — Football Player Tracking & Analytics System")
    print(f"  Server starting at: http://localhost:{port}")
    print(
        f"  Auto-reload: {'ON (dev)' if reload else 'off'} | Access gate: {'ON' if gated else 'off'}"
    )
    print("=" * 70)
    # ONE worker, always: JOBS, the in-flight-slug guard and chunked-upload state are in-process.
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=port, reload=reload, workers=1)
