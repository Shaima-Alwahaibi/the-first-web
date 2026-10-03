"""Rebuild the rule-based outputs and write the website-ready Phase 1 pack.

The rebuild reads the prepared historical files and the original rule workbook.
It does not write those sources.
"""

from __future__ import annotations

import pandas as pd

from advising_rule_engine import build_advising_output, export_advising_results
from course_difficulty_analysis import build_difficulty_output, export_difficulty_results
from prerequisite_checker import build_prerequisite_output, export_prerequisite_results
from project_paths import OUTPUT_DIR, REFERENCE_WORKBOOK, TRANSCRIPT_WORKBOOK
from repeat_classification import derive_repeat_classification
from rule_based_recommender import build_rule_based_recommendations, export_rule_based_recommendations
from study_plan_matching import HOLDOUT_TOKEN

STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
STATUS_WITHDRAWN = "Withdrawn"
REPEAT_COLUMNS = [
    "Student Code", "Course Code", "Semester", "Attempt Number", "Grade", "Result",
    "Derived Repeat Type", "Previous Attempt Outcome", "Repeat Evidence", "Repeat Resolution Status",
    "Is Repeated Due To Failure", "Is Repeated For Improvement",
]

PACK_PATH = OUTPUT_DIR / "phase1_student_academic_pack.xlsx"


def rebuild_rule_outputs() -> dict[str, object]:
    """Rerun prerequisite checking, advising, difficulty labels, and recommendations."""

    reference = pd.read_excel(REFERENCE_WORKBOOK, sheet_name=None)
    historical_path = OUTPUT_DIR / "historical_data_before_spring_2026.xlsx"
    remaining_path = OUTPUT_DIR / "remaining_courses_by_student.xlsx"
    historical = pd.read_excel(historical_path, sheet_name="Transcript Extracted")
    remaining = pd.read_excel(remaining_path, sheet_name="Remaining Courses")
    remaining_summary = pd.read_excel(remaining_path, sheet_name="Student Remaining Summary")
    electives = pd.read_excel(remaining_path, sheet_name="Elective Requirements")
    profiles = pd.read_excel(OUTPUT_DIR / "student_academic_profiles.xlsx", sheet_name="Student Academic Profiles")
    original = pd.read_excel(TRANSCRIPT_WORKBOOK, sheet_name="Transcript Extracted")
    excluded = pd.read_excel(OUTPUT_DIR / "course_scope_preparation.xlsx", sheet_name="Excluded History")
    prerequisite = build_prerequisite_output(
        remaining,
        historical,
        reference["Study Plan Courses"],
        reference["Prerequisite Rules"],
        reference["Elective Pools"],
        reference["Advising Rules"],
        reference["Validation Lists"],
        profiles,
        original,
        excluded,
    )
    export_prerequisite_results(prerequisite, OUTPUT_DIR / "prerequisite_check_results.xlsx")
    advising = build_advising_output(
        profiles,
        remaining,
        prerequisite["results"],
        reference["Study Plan Courses"],
        reference["Advising Rules"],
        historical,
        mapping=pd.read_excel(OUTPUT_DIR / "student_study_plan_mapping.xlsx"),
        resolution=pd.read_excel(OUTPUT_DIR / "study_plan_mapping_resolution.xlsx"),
        remaining_summary=remaining_summary,
        electives=electives,
    )
    export_advising_results(advising, OUTPUT_DIR / "advising_rule_engine_results.xlsx")
    holdout = pd.read_excel(OUTPUT_DIR / "spring_2026_holdout.xlsx")
    difficulty = build_difficulty_output(
        original,
        reference["Study Plan Courses"],
        reference["Elective Pools"],
        historical_reference_rows=len(historical),
        holdout_reference_rows=len(holdout),
    )
    export_difficulty_results(difficulty, OUTPUT_DIR / "course_difficulty_analysis.xlsx")
    recommendations = build_rule_based_recommendations(
        advising["state"],
        advising["candidates"],
        remaining,
        remaining_summary,
        difficulty["features"],
    )
    export_rule_based_recommendations(recommendations, OUTPUT_DIR / "rule_based_recommendations.xlsx")
    pack = build_phase1_student_pack(
        profiles=profiles,
        course_history=pd.read_excel(OUTPUT_DIR / "student_academic_profiles.xlsx", sheet_name="Course History Status"),
        remaining=remaining,
        remaining_summary=remaining_summary,
        prerequisites=prerequisite["results"],
        candidates=advising["candidates"],
        recommendations=recommendations["students"],
        selected=recommendations["recommendations"],
        historical=historical,
        pools=reference["Elective Pools"],
    )
    _write_pack(pack, PACK_PATH)
    return {
        "prerequisite_validation": prerequisite["validation"],
        "advising_validation": advising["validation"],
        "difficulty_validation": difficulty["validation"],
        "recommendation_validation": recommendations["validation"],
        "spring_leakage": {
            "prerequisite": prerequisite["spring_2026_leakage"],
            "advising": advising["spring_2026_leakage"],
            "difficulty": difficulty["spring_2026_rows_in_historical_features"],
        },
    }


def build_phase1_student_pack(
    *,
    profiles: pd.DataFrame,
    course_history: pd.DataFrame,
    remaining: pd.DataFrame,
    remaining_summary: pd.DataFrame,
    prerequisites: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
    selected: pd.DataFrame,
    historical: pd.DataFrame,
    pools: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Assemble the twelve student-level views and the named elective candidate list."""

    history = course_history.copy()
    final = history["Final Status"].astype(str) if "Final Status" in history.columns else pd.Series("", index=history.index)
    eligible = candidates.loc[candidates["Candidate Status"].eq("Allowed")].copy()
    conditional = candidates.loc[candidates["Candidate Status"].isin(["Conditional Pathway", "Conditional"])].copy()
    not_eligible = candidates.loc[candidates["Candidate Status"].isin(["Blocked by Prerequisite", "Not Eligible"])].copy()
    warnings = candidates.loc[candidates["Candidate Status"].isin(["Manual Review", "Advisor Approval Required"])].copy()
    load = recommendations.loc[:, [
        column for column in [
            "Student Code", "Probation Status", "Minimum Courses", "Maximum Courses",
            "Minimum Credits", "Maximum Credits", "Confirmed Recommended Course Count",
            "Confirmed Recommended Credits", "Recommendation Status", "Warning",
        ] if column in recommendations.columns
    ]].copy()
    repeats = _repeated_courses(_without_spring_2026(historical))
    return {
        "Academic Summary": profiles,
        "Completed Courses": history.loc[final.eq(STATUS_COMPLETED)].copy(),
        "Failed Courses": history.loc[final.eq(STATUS_FAILED)].copy(),
        "Withdrawn Courses": history.loc[final.eq(STATUS_WITHDRAWN)].copy(),
        "Repeated Courses": repeats,
        "Remaining Requirements": remaining,
        "Eligible Courses": eligible,
        "Not Eligible Courses": not_eligible,
        "Conditional Courses": conditional,
        "Recommended Plan": selected,
        "Load Validation": load,
        "Advisor Warnings": warnings,
        "Elective Candidates": _elective_candidates(remaining, prerequisites, candidates, pools),
        "Remaining Summary": remaining_summary,
    }


def _elective_candidates(
    remaining: pd.DataFrame,
    prerequisites: pd.DataFrame,
    candidates: pd.DataFrame,
    pools: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Student Code", "Elective Pool", "Course Code", "Course Title", "Prerequisite Rule",
        "Prerequisite Status", "Evidence", "Candidate Status", "Reason", "Warning",
        "Conditional Pathway", "Candidate Scope",
    ]
    if remaining.empty or "Is Elective" not in remaining.columns or "Elective Pool" not in remaining.columns:
        return pd.DataFrame(columns=columns)
    electives = _pathway_key(remaining.loc[remaining["Is Elective"].astype(str).eq("Yes")].copy())
    electives = electives.loc[electives["Course Code"].map(lambda value: str(value).strip() not in {"", "nan"})].copy()
    if electives.empty:
        return pd.DataFrame(columns=columns)
    rule_lookup = pools.loc[:, ["Elective Pool", "Course Code", "Prerequisite Rule"]].drop_duplicates()
    checked = _pathway_key(prerequisites.loc[:, [
        column for column in [
            "Student Code", "Course Code", "Conditional Pathway", "Prerequisite Rule Original",
            "Eligibility Status", "Eligibility Evidence Status", "Eligibility Reason", "Candidate Scope",
        ] if column in prerequisites.columns
    ]].copy())
    advised = _pathway_key(candidates.loc[:, [
        column for column in ["Student Code", "Course Code", "Conditional Pathway", "Candidate Status", "Candidate Reason"]
        if column in candidates.columns
    ]].copy())
    merged = electives.merge(rule_lookup, on=["Elective Pool", "Course Code"], how="left")
    merged = merged.merge(checked, on=["Student Code", "Course Code", "Conditional Pathway"], how="left")
    merged = merged.merge(advised, on=["Student Code", "Course Code", "Conditional Pathway"], how="left")
    merged["Prerequisite Rule"] = merged["Prerequisite Rule"].where(merged["Prerequisite Rule"].notna(), merged.get("Prerequisite Rule Original"))
    merged["Warning"] = merged["Candidate Status"].map(
        lambda status: "" if str(status) == "Allowed" else "Not a confirmed recommendation until the listed condition is cleared."
    )
    source_columns = [
        "Student Code", "Elective Pool", "Course Code", "Course Title", "Prerequisite Rule",
        "Eligibility Status", "Eligibility Evidence Status", "Candidate Status", "Candidate Reason",
        "Warning", "Conditional Pathway", "Candidate Scope",
    ]
    frame = merged.loc[:, [column for column in source_columns if column in merged.columns]].copy()
    frame = frame.rename(columns={
        "Eligibility Status": "Prerequisite Status",
        "Eligibility Evidence Status": "Evidence",
        "Candidate Reason": "Reason",
    })
    return frame.drop_duplicates(["Student Code", "Elective Pool", "Course Code", "Conditional Pathway"])


def _repeated_courses(historical: pd.DataFrame) -> pd.DataFrame:
    """Classify repeats from the previous attempt. Spring 2026 rows are already removed."""

    if historical is None or historical.empty:
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    classified = derive_repeat_classification(historical)
    if classified.empty or "Derived Repeat Type" not in classified.columns:
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    repeated = classified.loc[classified["Derived Repeat Type"].astype(str).ne("")].copy()
    return repeated.loc[:, [column for column in REPEAT_COLUMNS if column in repeated.columns]].copy()


def _without_spring_2026(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop Spring 2026 holdout rows before any pre-holdout pack sheet is built."""

    if frame is None or frame.empty:
        return frame
    semester = frame["Semester"].astype(str) if "Semester" in frame.columns else pd.Series("", index=frame.index)
    year = frame["Academic Year"].astype(str) if "Academic Year" in frame.columns else pd.Series("", index=frame.index)
    term = frame["Term"].astype(str) if "Term" in frame.columns else pd.Series("", index=frame.index)
    holdout = semester.str.contains(HOLDOUT_TOKEN, na=False) | year.eq(HOLDOUT_TOKEN) | (
        year.str.contains(HOLDOUT_TOKEN, na=False) & term.str.contains("Spring", case=False, na=False)
    )
    return frame.loc[~holdout].copy()


def _pathway_key(frame: pd.DataFrame) -> pd.DataFrame:
    if "Conditional Pathway" not in frame.columns:
        frame["Conditional Pathway"] = ""
    frame["Conditional Pathway"] = frame["Conditional Pathway"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    return frame


def _write_pack(tables: dict[str, pd.DataFrame], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
