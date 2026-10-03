"""Build in-memory Excel and CSV exports from a recommendation result."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from app.recommendation_service import RecommendationResult


def recommendation_workbook(result: RecommendationResult) -> bytes:
    """Return the four manager sheets: summary, eligible list, semester plan, blocked courses."""

    profile = pd.DataFrame([{"Field": key, "Value": value} for key, value in result.student_profile.items()])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        profile.to_excel(writer, sheet_name="Student Summary", index=False)
        _view(
            result.eligible_courses,
            {"Course": "Course Code"},
            ["Priority", "Course Code", "Course Name", "Type", "Credits", "Rule Score", "Reason"],
        ).to_excel(writer, sheet_name="All Eligible Courses", index=False)
        _view(
            result.selected_courses,
            {"Course": "Course Code", "Credit Hours": "Credits", "Recommendation Reason": "Reason"},
            ["Rank", "Course Code", "Course Name", "Type", "Credits", "Rule Score", "Reason"],
        ).to_excel(writer, sheet_name="Recommended Semester Plan", index=False)
        _view(
            result.blocked_courses,
            {"Course": "Course Code", "Score": "Rule Score"},
            ["Course Code", "Course Name", "Rule Score", "Reason"],
        ).to_excel(writer, sheet_name="Blocked Courses", index=False)
    return buffer.getvalue()


def _view(frame: pd.DataFrame, renames: dict[str, str], columns: list[str]) -> pd.DataFrame:
    view = frame.rename(columns=renames)
    present = [column for column in columns if column in view.columns]
    if not present:
        return pd.DataFrame(columns=columns)
    return view.loc[:, present]
