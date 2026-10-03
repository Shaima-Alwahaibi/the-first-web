"""Build in-memory Excel and CSV exports from a recommendation result."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from app.recommendation_service import RecommendationResult


def recommendation_workbook(result: RecommendationResult) -> bytes:
    """Return a three-sheet Excel file for the manager demo."""

    profile = pd.DataFrame([{"Field": key, "Value": value} for key, value in result.student_profile.items()])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        profile.to_excel(writer, sheet_name="Student Summary", index=False)
        result.selected_courses.to_excel(writer, sheet_name="Recommended Courses", index=False)
        result.blocked_courses.to_excel(writer, sheet_name="Blocked Courses", index=False)
    return buffer.getvalue()
