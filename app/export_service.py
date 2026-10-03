"""Build in-memory Excel and CSV exports from a recommendation result."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from app.recommendation_service import RecommendationResult


def recommendation_workbook(result: RecommendationResult) -> bytes:
    """Return an Excel file with the manager-facing Phase-1 sheets."""

    profile = pd.DataFrame([{"Field": key, "Value": value} for key, value in result.student_profile.items()])
    history = _history(result)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        profile.to_excel(writer, sheet_name="Student Summary", index=False)
        history.to_excel(writer, sheet_name="Course History", index=False)
        result.remaining_courses.to_excel(writer, sheet_name="Remaining Courses", index=False)
        result.selected_courses.to_excel(writer, sheet_name="Recommended Courses", index=False)
        result.blocked_courses.to_excel(writer, sheet_name="Blocked Courses", index=False)
        result.prerequisite_results.to_excel(writer, sheet_name="Prerequisite Results", index=False)
        pd.DataFrame({"Warning": result.warnings or ["None"]}).to_excel(writer, sheet_name="Warnings", index=False)
    return buffer.getvalue()


def recommended_csv(result: RecommendationResult) -> bytes:
    """Return the confirmed recommendation table as CSV."""

    return result.selected_courses.to_csv(index=False).encode("utf-8")


def _history(result: RecommendationResult) -> pd.DataFrame:
    frames = []
    for label, frame in result.course_history.items():
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        copy.insert(0, "History Section", label)
        frames.append(copy)
    if not frames:
        return pd.DataFrame(columns=["History Section"])
    return pd.concat(frames, ignore_index=True, sort=False)
