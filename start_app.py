#!/usr/bin/env python3
"""Unified Launcher for AI Football Player Tracking & Analytics Web Server."""

import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("=" * 70)
    print("  PITCHVISION AI — Football Player Tracking & Analytics System")
    print("  Server starting at: http://localhost:8000")
    print("=" * 70)
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
