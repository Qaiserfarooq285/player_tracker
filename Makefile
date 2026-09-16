.PHONY: setup install-gpu models serve run eval test lint format clean-work profile

PYTHON := .venv/bin/python
PIP    := .venv/bin/python -m pip
UV     := $$HOME/.local/bin/uv

# Create the venv and install core + dev deps (CLAUDE.md §8). Does NOT pull torch — that's
# `make install-gpu`, kept separate so `make setup` stays fast and lean.
setup:
	export PATH="$$HOME/.local/bin:$$PATH" && $(UV) venv --python 3.11
	export PATH="$$HOME/.local/bin:$$PATH" && $(UV) pip install -e ".[dev]"

# Adds the heavy, GPU-bound extras (RF-DETR/torch detection stack + SigLIP team stack) on top of
# `make setup`, pinned to the CUDA 12.1 torch wheel index (RTX A2000, CLAUDE.md §11).
install-gpu:
	export PATH="$$HOME/.local/bin:$$PATH" && $(UV) pip install -e ".[detect,team,ocr,jersey_parseq]" \
		--extra-index-url https://download.pytorch.org/whl/cu121

# Fetch + checksum-verify the three model checkpoints into models/ (no-op once present).
# Also what docker/runpod_bootstrap.sh runs on every pod boot (docs/DEPLOY.md).
models:
	$(PYTHON) scripts/download_models.py

# Web UI + API on http://localhost:8000. PV_DEV=1 turns on uvicorn auto-reload for local editing
# (never on a hosted pod: it kills in-flight jobs). PV_ACCESS_PASSWORD=... enables the login gate.
serve:
	$(PYTHON) start_app.py

# Process the video(s) in input/ -> reel + stat card + report in output/ (resumable via work/).
# CLAUDE.md §14: auto-branches per video (manual-annotation sidecar -> filename jersey -> ADR-15
# auto). VIDEO=<path> targets a single video instead of every file in input/; TARGET=<n> passes a
# human-confirmed jersey number for a filename-less video (ADR-18's --target-jersey seam) --
# meaningless (and harmlessly ignored by main()) for a video with a filename-parsed number.
run:
	$(PYTHON) -m src.pipeline.run $(if $(VIDEO),"$(VIDEO)") $(if $(TARGET),--target-jersey $(TARGET))

# Detection mAP + tracking HOTA (+ action mAP@1 later) on data/eval/ (CLAUDE.md §9).
eval:
	$(PYTHON) -m src.eval.run

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m black .

# Wipe cached, resumable per-stage artifacts (work/ is gitignored and safe to nuke and rebuild).
clean-work:
	rm -rf work/*
	@mkdir -p work && touch work/.gitkeep

# Run the Stage 0.5 source profiler over every video in input/ (CLAUDE.md §5 Stage 0.5).
profile:
	$(PYTHON) -m src.pipeline.profile_cli input
