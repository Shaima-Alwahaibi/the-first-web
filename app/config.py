"""Paths, limits, and user-facing messages for the Phase-1 demo.

Academic rule scores are not defined here. They are read from the validated
rule-based recommender.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1]
PHASE1_MODULE_NAMES = (
    "academic_evidence.py",
    "advising_rule_engine.py",
    "course_difficulty_analysis.py",
    "phase1_student_pack.py",
    "preparation_checks.py",
    "prerequisite_checker.py",
    "project_paths.py",
    "remaining_courses.py",
    "repeat_classification.py",
    "rule_based_recommender.py",
    "student_profiles.py",
    "study_plan_matching.py",
)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
HOLD_OUT_LABEL = "Spring 2026"

MSG_UNREADABLE = (
    "The transcript could not be read. Please verify that this is a supported UTAS transcript PDF."
)
MSG_IMAGE = (
    "This transcript appears to be image-based or unsupported by the current Phase-1 parser."
)
MSG_EMPTY = "The uploaded file is empty."
MSG_TYPE = "Please upload a PDF transcript (.pdf)."
MSG_SIZE = "The transcript is larger than the 8 MB limit for this demo."
MSG_SPECIALIZATION = (
    "Specialization could not be determined automatically. "
    "Academic recommendation cannot safely continue without study-plan identification."
)
MSG_PREREQUISITE = (
    "Recommendation requires academic review because prerequisite information is incomplete."
)
MSG_PRIVACY = (
    "Uploaded transcripts are processed for the current session and are not "
    "intentionally stored permanently by this Phase-1 demo."
)

UNRESOLVED_PATHWAYS = {
    "",
    "manual review",
    "pathway unknown",
    "pathway resolution required",
    "not available before spring 2026",
}


def research_root() -> Path | None:
    """Return the research project that contains the validated Phase-1 modules."""

    configured = os.environ.get("PHASE1_PROJECT_ROOT", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(WEB_ROOT.parent)
    candidates.append(WEB_ROOT / "phase1_modules")
    for candidate in candidates:
        if (candidate / "rule_based_recommender.py").is_file():
            return candidate.resolve()
    return None


def ensure_research_importable() -> Path:
    """Put the validated research modules on sys.path and return their directory."""

    root = research_root()
    if root is None:
        raise RuntimeError(
            "The Phase-1 research modules could not be found. "
            "Set PHASE1_PROJECT_ROOT to the research project directory."
        )
    location = str(root)
    if location not in sys.path:
        sys.path.insert(0, location)
    return root


def output_dir() -> Path:
    """Prefer the research outputs. Fall back to the bundled demo extract."""

    root = research_root()
    if root is not None:
        parent_outputs = root / "outputs"
        if (parent_outputs / "phase1_student_academic_pack.xlsx").is_file():
            return parent_outputs
    bundled = WEB_ROOT / "data" / "demo"
    if (bundled / "phase1_student_academic_pack.xlsx").is_file():
        return bundled
    raise FileNotFoundError(
        "Validated Phase-1 outputs are not available. The demo cannot build a new academic result."
    )


def sample_pdfs() -> dict[str, Path]:
    """Return the real sample transcripts shipped with the research project."""

    names = {
        "STUD-175": "sample_transcript_STUD-175.pdf",
        "STUD-016": "sample_transcript_STUD-016.pdf",
    }
    search = [
        WEB_ROOT / "data" / "samples",
        WEB_ROOT.parent / "data" / "raw" / "samples",
    ]
    found: dict[str, Path] = {}
    for student_code, filename in names.items():
        for folder in search:
            path = folder / filename
            if path.is_file():
                found[student_code] = path
                break
    return found
