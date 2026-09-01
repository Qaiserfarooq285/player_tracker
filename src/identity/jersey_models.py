"""Shared, fail-soft loader for the two owner-authorized jersey-reading checkpoints (CLAUDE.md §7
"mkoshkina/jersey-number-pipeline CHECKPOINTS ONLY" row) -- used by BOTH `src/identity/verify.py`
(ADR-15's per-take verification) and `src/annotations/associate.py`/`src/pipeline/manual_events.py`
(ADR-19's manual-mode jersey re-identification), so the two callers share one loading/degradation
policy instead of two divergent copies.

**Fail-soft by design.** These two checkpoints are optional and NC-restricted: a repo checkout
without `models/jersey-parseq-soccernet/*` populated (e.g. a fresh clone, or a deliberate choice
to avoid the CC BY-NC 3.0 encumbrance, `configs/identity.yaml: legibility.enabled: false`/
`parseq_soccernet.enabled: false`) must keep working exactly as it did before this feature existed
-- the pre-existing EasyOCR/Gemini chain. `load_optional_jersey_stack` therefore NEVER raises for
"missing file"/"disabled in config"/"import error" -- it logs a warning and returns `(None, None,
None)`, which every caller already treats as "this new signal did not run" (see
`src/identity/verify.py::_collect_reads_for_take`'s own `legibility_model is not None and
parseq_model is not None` gate). A genuine ARCHITECTURE mismatch inside `load_parseq_soccernet`
itself (a checkpoint that doesn't fit the verified remap) still raises loudly from THAT function --
this wrapper only catches the "not available at all" cases, never silently swallows a real bug in
a checkpoint that IS present and enabled.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.identity.jersey_parseq import free_parseq_soccernet, load_parseq_soccernet
from src.identity.legibility import free_legibility_model, load_legibility_model

logger = get_logger(__name__)


def load_optional_jersey_stack(
    identity_cfg: dict, device: str = "cuda"
) -> tuple[Any, Any, Any]:
    """Returns `(legibility_model, parseq_model, parseq_transform)`, each `None` if that stage is
    disabled or its checkpoint can't be loaded (missing file, bad checkpoint, no `identity_cfg`
    entry for it at all) -- see module docstring for why this never raises.
    """
    legibility_cfg = identity_cfg.get("legibility") or {}
    parseq_cfg = identity_cfg.get("parseq_soccernet") or {}

    legibility_model = None
    if legibility_cfg.get("enabled", True) and legibility_cfg.get("checkpoint"):
        try:
            legibility_model = load_legibility_model(legibility_cfg["checkpoint"], device=device)
        except Exception:
            logger.warning(
                "legibility checkpoint %r could not be loaded -- jersey reads fall back to the "
                "pre-existing EasyOCR/Gemini chain unchanged (this is not fatal)",
                legibility_cfg.get("checkpoint"),
                exc_info=True,
            )
            legibility_model = None

    parseq_model = None
    parseq_transform = None
    if parseq_cfg.get("enabled", True) and parseq_cfg.get("checkpoint"):
        try:
            parseq_model, parseq_transform = load_parseq_soccernet(
                parseq_cfg["checkpoint"], device=device
            )
        except Exception:
            logger.warning(
                "PARSeq-SoccerNet checkpoint %r could not be loaded -- jersey reads fall back to "
                "the pre-existing EasyOCR/Gemini chain unchanged (this is not fatal)",
                parseq_cfg.get("checkpoint"),
                exc_info=True,
            )
            parseq_model = None
            parseq_transform = None

    # The chain requires BOTH models to attempt the new path at all (`_collect_reads_for_take`'s
    # own gate is `legibility_model is not None and parseq_model is not None`) -- if only one
    # loaded, free it immediately rather than holding VRAM for a stage that can never fire.
    if legibility_model is not None and parseq_model is None:
        free_legibility_model(legibility_model)
        legibility_model = None
    elif parseq_model is not None and legibility_model is None:
        free_parseq_soccernet(parseq_model)
        parseq_model, parseq_transform = None, None

    return legibility_model, parseq_model, parseq_transform


def free_optional_jersey_stack(legibility_model: Any, parseq_model: Any) -> None:
    """Free whichever of the two optional models is non-`None` (CLAUDE.md §11)."""
    if legibility_model is not None:
        free_legibility_model(legibility_model)
    if parseq_model is not None:
        free_parseq_soccernet(parseq_model)
