"""Read validated Phase-1 outputs for one student.

The adapter does not recalculate eligibility or rule scores. Those values are
taken from the workbooks produced by the existing research modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import MSG_PREREQUISITE, MSG_SPECIALIZATION, UNRESOLVED_PATHWAYS, output_dir

PACK_NAME = "phase1_student_academic_pack.xlsx"
RECOMMENDATION_NAME = "rule_based_recommendations.xlsx"
PREREQUISITE_NAME = "prerequisite_check_results.xlsx"
HOLDOUT_FILE = "spring_2026_holdout.xlsx"


class PipelineDataError(Exception):
    """Validated Phase-1 data cannot support a safe recommendation."""


def load_student(student_code: str) -> dict[str, object]:
    """Return the validated profile, history, eligibility, and recommendation rows."""

    code = str(student_code).strip()
    catalog = _catalog()
    profiles = _filter(catalog["profiles"], code)
    if profiles.empty:
        raise PipelineDataError(MSG_SPECIALIZATION)
    recommendations = _filter(catalog["recommendations"], code)
    if recommendations.empty:
        raise PipelineDataError(MSG_SPECIALIZATION)
    pathway = _text(recommendations.iloc[0].get("Assigned Pathway"))
    readiness = _text(recommendations.iloc[0].get("Pathway Readiness"))
    status = _text(recommendations.iloc[0].get("Recommendation Status"))
    if _unresolved(pathway) or _unresolved(readiness) or _unresolved(status):
        raise PipelineDataError(MSG_SPECIALIZATION)
    prerequisites = _filter(catalog["prerequisites"], code)
    if catalog["prerequisites"].empty:
        raise PipelineDataError(MSG_PREREQUISITE)
    return {
        "profile": profiles.iloc[0].to_dict(),
        "summary": _one(catalog["remaining_summary"], code),
        "load": _one(catalog["load"], code),
        "recommendation": recommendations.iloc[0].to_dict(),
        "completed": _filter(catalog["completed"], code),
        "failed": _filter(catalog["failed"], code),
        "withdrawn": _filter(catalog["withdrawn"], code),
        "repeated": _filter(catalog["repeated"], code),
        "remaining": _filter(catalog["remaining"], code),
        "selected": _filter(catalog["selected"], code),
        "audit": _filter(catalog["audit"], code),
        "prerequisites": prerequisites,
        "eligible": _filter(catalog["eligible"], code),
        "warnings": _filter(catalog["advisor_warnings"], code),
    }


def materialize_demo_catalog(students: tuple[str, ...] = ("STUD-016", "STUD-175")) -> Path:
    """Write a two-student extract so a clean clone can run the sample demo."""

    from app.config import WEB_ROOT

    destination = output_dir()
    if destination.name == "demo":
        return destination
    demo = WEB_ROOT / "data" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    catalog = _read_directory(destination)
    codes = set(students)

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "Student Code" not in frame.columns:
            return frame.copy()
        return frame.loc[frame["Student Code"].astype(str).isin(codes)].copy()

    pack_sheets = {
        "Academic Summary": keep(catalog["profiles"]),
        "Completed Courses": keep(catalog["completed"]),
        "Failed Courses": keep(catalog["failed"]),
        "Withdrawn Courses": keep(catalog["withdrawn"]),
        "Repeated Courses": keep(catalog["repeated"]),
        "Remaining Requirements": keep(catalog["remaining"]),
        "Eligible Courses": keep(catalog["eligible"]),
        "Recommended Plan": keep(catalog["selected"]),
        "Load Validation": keep(catalog["load"]),
        "Advisor Warnings": keep(catalog["advisor_warnings"]),
        "Remaining Summary": keep(catalog["remaining_summary"]),
    }
    _write_sheets(demo / PACK_NAME, pack_sheets)
    _write_sheets(
        demo / RECOMMENDATION_NAME,
        {
            "Student Recommendations": keep(catalog["recommendations"]),
            "Recommended Courses": keep(catalog["selected"]),
            "Candidate Ranking Audit": keep(catalog["audit"]),
        },
    )
    _write_sheets(demo / PREREQUISITE_NAME, {"Prerequisite Results": keep(catalog["prerequisites"])})
    return demo


@lru_cache(maxsize=1)
def _catalog() -> dict[str, pd.DataFrame]:
    folder = output_dir()
    if folder.name != "outputs" and (folder / HOLDOUT_FILE).is_file():
        raise PipelineDataError("The demo catalog must not include the Spring 2026 holdout workbook.")
    return _read_directory(folder)


def _read_directory(folder: Path) -> dict[str, pd.DataFrame]:
    pack = _sheets(folder / PACK_NAME)
    recommendations = _sheets(folder / RECOMMENDATION_NAME)
    prerequisites = _sheets(folder / PREREQUISITE_NAME)
    required = ("Academic Summary", "Recommended Plan", "Remaining Requirements")
    missing = [name for name in required if name not in pack]
    if missing or "Student Recommendations" not in recommendations or "Candidate Ranking Audit" not in recommendations:
        raise PipelineDataError("A validated Phase-1 workbook is missing a required sheet.")
    if "Prerequisite Results" not in prerequisites:
        raise PipelineDataError(MSG_PREREQUISITE)
    return {
        "profiles": pack["Academic Summary"],
        "completed": pack.get("Completed Courses", pd.DataFrame()),
        "failed": pack.get("Failed Courses", pd.DataFrame()),
        "withdrawn": pack.get("Withdrawn Courses", pd.DataFrame()),
        "repeated": pack.get("Repeated Courses", pd.DataFrame()),
        "remaining": pack["Remaining Requirements"],
        "remaining_summary": pack.get("Remaining Summary", pd.DataFrame()),
        "selected": pack["Recommended Plan"],
        "load": pack.get("Load Validation", pd.DataFrame()),
        "eligible": pack.get("Eligible Courses", pd.DataFrame()),
        "advisor_warnings": pack.get("Advisor Warnings", pd.DataFrame()),
        "recommendations": recommendations["Student Recommendations"],
        "audit": recommendations["Candidate Ranking Audit"],
        "prerequisites": prerequisites["Prerequisite Results"],
    }


def _sheets(path: Path) -> dict[str, pd.DataFrame]:
    if not path.is_file():
        raise PipelineDataError(f"Required Phase-1 workbook is unavailable: {path.name}")
    return pd.read_excel(path, sheet_name=None)


def _write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def _filter(frame: pd.DataFrame, student_code: str) -> pd.DataFrame:
    if frame is None or frame.empty or "Student Code" not in frame.columns:
        return pd.DataFrame() if frame is None else frame.iloc[0:0].copy()
    return frame.loc[frame["Student Code"].astype(str).eq(student_code)].copy()


def _one(frame: pd.DataFrame, student_code: str) -> dict[str, object]:
    filtered = _filter(frame, student_code)
    if filtered.empty:
        return {}
    return filtered.iloc[0].to_dict()


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _unresolved(value: str) -> bool:
    return value.casefold() in UNRESOLVED_PATHWAYS
