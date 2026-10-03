"""Check extracted transcript fields before any recommendation is requested.

Validation reports extraction quality. It does not decide academic eligibility.
"""

from __future__ import annotations

import pandas as pd

from app.config import MSG_UNREADABLE


def validate_transcript(parsed: dict[str, object]) -> dict[str, list[str]]:
    """Return blocking errors and non-blocking extraction warnings."""

    errors: list[str] = []
    warnings = [str(item) for item in parsed.get("warnings", [])]
    student = parsed.get("student") or {}
    attempts = parsed.get("course_attempts")
    if not isinstance(student, dict) or not str(student.get("student_code", "")).strip():
        errors.append("Student code was not found, so the transcript cannot be matched safely.")
    if not isinstance(attempts, pd.DataFrame) or attempts.empty:
        errors.append(MSG_UNREADABLE)
    else:
        codes = attempts["Course Code"].astype(str).str.strip()
        if codes.eq("").any() or codes.eq("nan").any():
            errors.append("One or more extracted rows do not have a course code.")
        if attempts["Grade"].isna().all() and attempts["Result"].isna().all():
            errors.append("Grades and results could not be read from the transcript.")
    return {"errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings))}
