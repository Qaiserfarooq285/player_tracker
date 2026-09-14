"""Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14) — a real, downloadable PDF rendering
of the owner's `statcard.md` template (CLAUDE.md §13.2).

**Why this exists.** The frontend's own hand-rolled exporter (`apps/web/js/app.js::exportStatCard`,
not this session's file) re-types an approximation of the statcard client-side and, because its
`_safe_int` coercion (`apps/api/main.py::_safe_int`) turns the non-numeric `"not available (...)"`
string into a bare `0`, it prints **`Goals: 0`** where the authoritative `statcard.md` carries the
full "not available" explanation — a Golden Rule 5 ("no fabricated numbers") violation in shipped
code. `render_statcard_pdf` is the honest replacement: a server-rendered PDF generated from the
SAME real values `render_statcard_markdown` uses, so there is no second, drifting re-typing of the
stats anywhere.

**Same input tuple, on purpose.** `render_statcard_pdf` takes the exact same positional argument
tuple as `render_statcard_markdown` (`src/pipeline/player_output.py`), plus a required
keyword-only `output_path` — see that function's own docstring for what each argument means
(`goal_reason`/`identity_status` in particular). Keeping the tuple identical is what stops the two
renderers from silently diverging over time; where their *goal/assist line* logic
(count-vs-`goal_reason`) has to be reimplemented here (a PDF flowable is not a markdown string), it
mirrors `render_statcard_markdown`'s own logic line-for-line — if that logic ever changes, this
copy must change with it in the same commit.

**Optional dependency, fail-soft.** `reportlab` (BSD-3-Clause, CLAUDE.md §7, the `api` extra in
`pyproject.toml`) is imported LAZILY, inside `render_statcard_pdf` itself, never at module level —
same convention as `src/identity/jersey_ocr.py::load_easyocr_reader`'s lazy `import easyocr` and
`src/team/classifier.py::load_siglip`'s lazy `import transformers`. That keeps THIS module (and
anything that imports it, e.g. `src/pipeline/player_output.py`, which imports it unconditionally)
importable in an environment without the `api` extra installed. The caller
(`write_player_output`) is what catches the resulting `ImportError` and degrades to "skip the PDF,
keep the markdown" — see that function's own comment. This module never catches its own
`ImportError`; it is the caller's job, exactly like `load_optional_jersey_stack`
(`src/identity/jersey_models.py`) puts the try/except at the call site, not inside the loader.
"""

from __future__ import annotations

from pathlib import Path

from src.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------------------------
# Layout constants (CLAUDE.md §10: no magic numbers inline). Unlike this repo's measured
# detection/tracking thresholds, there is nothing to MEASURE for a text document's own page
# margins/font sizes -- these are plain, conventional print-layout choices -- but they are still
# named here rather than left as bare literals in the flowable calls below, matching the "config
# or named module-level constant" alternative this project's own conventions allow (see
# configs/target.yaml's own "REASONED-BUT-UNMEASURED" markers for values that similarly have no
# real measurement to anchor them, just a stated reason).
# ---------------------------------------------------------------------------------------------
PAGE_MARGIN_IN = 0.75  # inches, all four sides -- a conventional, comfortable print/read margin.
TITLE_FONT_SIZE = 18  # "Player Statistics" -- the document's single most prominent line.
SUBTITLE_FONT_SIZE = 13  # "Player #<N>" and "Identity Status".
BODY_FONT_SIZE = 10  # the stat block itself (Touches/Passes/.../Distance Covered).
SECTION_HEADING_FONT_SIZE = 13  # "Event Timeline".
TABLE_HEADER_FONT_SIZE = 9
TABLE_BODY_FONT_SIZE = 8.5  # slightly smaller than the stat block: the timeline can run to many
# rows, and CLAUDE.md's own honesty requirement (print "not available (...)" reasons verbatim,
# never clipped) means the Source/Event columns must accommodate long strings -- a smaller body
# font buys more per-line character budget before Paragraph word-wraps a cell.
SPACER_SMALL_PT = 6  # between adjacent stat lines
SPACER_MEDIUM_PT = 16  # between major sections (header block -> stat block -> timeline heading)
# Event Timeline column widths (inches). Four columns -- Time, Event, Confidence, Source -- sized
# so "Source" (the widest real-world strings this repo emits, e.g. "scoreboard_delta",
# "manual_annotation", "gemini_vlm") gets the most room, while still summing to comfortably under
# a US-Letter page's printable width (8.5in - 2*PAGE_MARGIN_IN = 7.0in here).
TIMELINE_COL_WIDTHS_IN = (0.95, 1.55, 0.95, 3.55)
# reportlab compresses each page's content stream (Flate) by default (`rl_settings.pageCompression
# = 1`). A statcard is a small, all-text document -- the size saving is negligible -- and leaving
# it uncompressed means the PDF's own text is a plain, greppable byte string (what this module's
# own anti-regression test, and anyone debugging a bad render by eye with `strings`/`less`, relies
# on), so compression is deliberately turned off rather than left at the library default.
PAGE_COMPRESSION = 0


def _format_timestamp(seconds: float) -> str:
    """`M:SS.s` rendering. Deliberately DUPLICATED from
    `src.pipeline.player_output._format_timestamp` (same one-line body) rather than imported --
    `player_output.py` already imports FROM this module (to call `render_statcard_pdf`), so
    importing back from `player_output` here would be a circular import. Same
    per-module-independence tradeoff this repo already makes elsewhere (e.g.
    `configs/target.yaml`'s own duplicated-not-cross-imported threshold values, see that file's
    header comment) -- if the format ever changes, both copies must change together.
    """
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:04.1f}"


def _goal_assist_text(count: int, goal_reason: str | None) -> str:
    """Mirrors `render_statcard_markdown`'s own goals_line/assists_line logic (CLAUDE.md Golden
    Rule 5): a non-`None` `goal_reason` with a zero count means the zero is genuinely UNCERTAIN (no
    scoreboard/goal-region/annotation source ran), so the reason text is shown verbatim instead of
    a bare `0` that would overclaim certainty; `goal_reason=None` means the run's event source is
    authoritative and a real `0` is the honest answer. See that function's own docstring for the
    full reasoning -- this must stay in sync with it.
    """
    return str(count) if count > 0 or goal_reason is None else goal_reason


def render_statcard_pdf(
    jersey_number: int,
    counts: dict[str, int],
    possession_seconds: float | None,
    distance_result: dict | None,
    timeline_rows: list[dict],
    goal_reason: str | None = None,
    identity_status: str = "Verified",
    *,
    output_path: Path,
) -> None:
    """Render the same real values `render_statcard_markdown` uses to a PDF at `output_path`.

    Raises `ImportError` if `reportlab` is not installed -- callers (`write_player_output`) are
    responsible for catching that and degrading to markdown-only (see module docstring). Any other
    exception is a genuine rendering bug and is allowed to propagate for the same reason
    `load_parseq_soccernet` lets an architecture-mismatch exception propagate past
    `load_optional_jersey_stack`'s own fail-soft wrapper: this module's job is "render correctly or
    say why not", not to silently swallow a real defect in front of the caller who's already
    prepared to log+skip.
    """
    # Lazy import -- see module docstring. Keeps this module (and player_output.py, which imports
    # it unconditionally) importable without the `api` extra's reportlab installed.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "StatcardTitle",
        parent=styles["Title"],
        fontSize=TITLE_FONT_SIZE,
        spaceAfter=SPACER_SMALL_PT,
    )
    subtitle_style = ParagraphStyle(
        "StatcardSubtitle", parent=styles["Normal"], fontSize=SUBTITLE_FONT_SIZE, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "StatcardBody",
        parent=styles["Normal"],
        fontSize=BODY_FONT_SIZE,
        leading=BODY_FONT_SIZE * 1.35,  # generous line spacing so a wrapped "not available (...)"
        # reason line stays readable rather than cramped against the line above/below it.
    )
    heading_style = ParagraphStyle(
        "StatcardHeading",
        parent=styles["Heading2"],
        fontSize=SECTION_HEADING_FONT_SIZE,
        spaceBefore=SPACER_MEDIUM_PT,
        spaceAfter=SPACER_SMALL_PT,
    )
    table_header_style = ParagraphStyle(
        "StatcardTableHeader",
        parent=styles["Normal"],
        fontSize=TABLE_HEADER_FONT_SIZE,
        fontName="Helvetica-Bold",
    )
    table_body_style = ParagraphStyle(
        "StatcardTableBody", parent=styles["Normal"], fontSize=TABLE_BODY_FONT_SIZE, leading=11
    )

    story: list = [
        Paragraph("Player Statistics", title_style),
        Paragraph(f"Player #{jersey_number}", subtitle_style),
        # Deliberately plain text, no inline <b>/<font> markup: a bold/normal run switch mid-line
        # makes reportlab emit the label and the value as SEPARATE Tj operators in the content
        # stream, which would break a byte-level substring check like "Goals: not available (...)"
        # -- exactly the kind of value this renderer exists to keep intact and verifiable (see
        # module docstring). A visual bold label is a cosmetic nicety, not worth that risk.
        Paragraph(f"Identity Status: {identity_status}", subtitle_style),
        Spacer(1, SPACER_SMALL_PT),
    ]

    goals_count = counts.get("goal", 0)
    assists_count = counts.get("assist", 0)
    goals_text = _goal_assist_text(goals_count, goal_reason)
    assists_text = _goal_assist_text(assists_count, goal_reason)
    possession_text = (
        f"{possession_seconds:.1f}s" if possession_seconds is not None else "uncertain"
    )
    distance_text = (
        f"{distance_result['distance']:.2f} {distance_result['unit']} (uncalibrated)"
        if distance_result is not None
        else "uncertain"
    )

    stat_lines = [
        ("Touches", counts.get("touch", 0)),
        ("Passes", counts.get("pass", 0)),
        ("Turnovers", counts.get("turnover", 0)),
        ("Sprints/Runs", counts.get("sprint", 0)),
        ("Goals", goals_text),
        ("Assists", assists_text),
        ("Shots", counts.get("shot", 0)),
        ("Tackles", counts.get("tackle", 0)),
        ("Saves", counts.get("save", 0)),
        ("Dribbles", counts.get("dribble", 0)),
        ("Possession Time", possession_text),
        ("Distance Covered", distance_text),
    ]
    for label, value in stat_lines:
        # Paragraph (not a plain drawString) so a long "not available (...)" reason wraps onto
        # further lines within the page instead of being clipped at the page edge -- the whole
        # point of this renderer over the frontend's old fixed-width text export.
        # Plain text, same reasoning as the Identity Status line above -- keeps "Goals: <value>"
        # (and every other stat line) as ONE contiguous drawn string, never split by a bold/normal
        # font-run boundary.
        story.append(Paragraph(f"{label}: {value}", body_style))

    story.append(Paragraph("Event Timeline", heading_style))

    if timeline_rows:
        header = [
            Paragraph("Time", table_header_style),
            Paragraph("Event", table_header_style),
            Paragraph("Confidence", table_header_style),
            Paragraph("Source", table_header_style),
        ]
        table_data = [header]
        for row in timeline_rows:
            table_data.append(
                [
                    Paragraph(_format_timestamp(row["t_start"]), table_body_style),
                    Paragraph(str(row["label"]), table_body_style),
                    Paragraph(f"{row['confidence']:.2f}", table_body_style),
                    # `source` (CLAUDE.md's own owner ask, Plan Stage 3 Context: "distinguishable
                    # from an auto-detected one") -- e.g. "manual_touch"/"manual_annotation" vs
                    # "possession_heuristic"/"scoreboard_delta"/"gemini_vlm". Never omitted.
                    Paragraph(str(row["source"]), table_body_style),
                ]
            )
        col_widths = [w * inch for w in TIMELINE_COL_WIDTHS_IN]
        timeline_table = Table(
            table_data,
            colWidths=col_widths,
            repeatRows=1,  # repeat the header row on every
            # page the table spills onto -- Table is a standard platypus flowable and splits
            # across page boundaries on its own when a long timeline doesn't fit on one page, no
            # extra pagination code needed here.
        )
        timeline_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(timeline_table)
    else:
        # Same "never pad the timeline to look complete" rule as render_statcard_markdown's own
        # "_(no events attributed to this player)_" row -- an empty timeline is a real, honest
        # answer, not an omission to paper over with a fabricated row.
        story.append(Paragraph("(no events attributed to this player)", body_style))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=PAGE_MARGIN_IN * inch,
        rightMargin=PAGE_MARGIN_IN * inch,
        topMargin=PAGE_MARGIN_IN * inch,
        bottomMargin=PAGE_MARGIN_IN * inch,
        pageCompression=PAGE_COMPRESSION,
        title=f"Player #{jersey_number} Statistics",
    )
    doc.build(story)
    logger.info("wrote statcard PDF: %s", output_path)
