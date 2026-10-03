"""Adapt a confirmed transcript to the validated Phase-1 recommendation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import HOLD_OUT_LABEL, MSG_PREREQUISITE, ensure_research_importable
from app.pipeline_adapter import PipelineDataError, load_student
from app.transcript_validator import validate_transcript

FUTURE_SCORE_FIELDS = (
    "grade_prediction_score",
    "cf_score",
    "content_score",
    "transformer_score",
    "cgpa_impact_score",
    "hybrid_score",
)


@dataclass
class RecommendationResult:
    """Phase-1 result. Only rule_score is populated; later scores stay unset."""

    student_profile: dict[str, object]
    course_history: dict[str, pd.DataFrame]
    remaining_courses: pd.DataFrame
    prerequisite_results: pd.DataFrame
    candidate_courses: pd.DataFrame
    eligible_courses: pd.DataFrame
    selected_courses: pd.DataFrame
    blocked_courses: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    stages: list[dict[str, str]] = field(default_factory=list)
    extraction: dict[str, object] = field(default_factory=dict)


def generate_recommendation(parsed: dict[str, object]) -> RecommendationResult:
    """Match a parsed transcript to the validated rule-based result for that student."""

    validation = validate_transcript(parsed)
    if validation["errors"]:
        raise PipelineDataError(validation["errors"][0])
    student = parsed.get("student")
    if not isinstance(student, dict):
        raise PipelineDataError("Student information was not extracted.")
    record = load_student(str(student.get("student_code", "")))
    return _assemble(parsed, record, validation["warnings"])


def rule_framework() -> pd.DataFrame:
    """Return the engine score table, including the non-selectable score of 0."""

    ensure_research_importable()
    from advising_rule_engine import STATUS_BLOCKED_PREREQUISITE
    from rule_based_recommender import CLASS_CURRENT, PRIORITY, REASONS, SCORE, _score

    rows = [
        {
            "Academic condition": label,
            "Rule Score": float(SCORE[label]),
            "Advisor meaning": REASONS[label],
        }
        for label in sorted(SCORE, key=lambda item: PRIORITY[item])
    ]
    rows.append(
        {
            "Academic condition": "Not academically eligible",
            "Rule Score": float(_score(CLASS_CURRENT, STATUS_BLOCKED_PREREQUISITE)),
            "Advisor meaning": (
                "Eligibility is decided before ranking. A course that is not selectable "
                "receives this score and is not confirmed."
            ),
        }
    )
    return pd.DataFrame(rows)


def _assemble(
    parsed: dict[str, object],
    record: dict[str, object],
    extraction_warnings: list[str],
) -> RecommendationResult:
    ensure_research_importable()
    from advising_rule_engine import (
        STATUS_ALLOWED,
        STATUS_BLOCKED_MIXING,
        STATUS_BLOCKED_PREREQUISITE,
        STATUS_CONDITIONAL,
        STATUS_MANUAL,
        STATUS_PATHWAY,
    )
    from prerequisite_checker import NOT_ELIGIBLE
    from rule_based_recommender import REASONS, STATUS_COMPLETE

    recommendation = record["recommendation"]
    profile = record["profile"]
    attempts = parsed.get("course_attempts")
    holdout_codes = set()
    if isinstance(attempts, pd.DataFrame) and not attempts.empty and "Holdout" in attempts.columns:
        holdout_codes = set(attempts.loc[attempts["Holdout"].eq(True), "Course Code"].astype(str))

    selected = _present_selected(record["selected"], REASONS, holdout_codes)
    eligible = _present_eligible(record["audit"], selected, REASONS, STATUS_ALLOWED)
    blocked = _present_blocked(record["audit"], STATUS_ALLOWED)
    prerequisites = record["prerequisites"]
    warnings = list(extraction_warnings)
    pdf_specialization = _text((parsed.get("student") or {}).get("specialization"))
    pathway = _text(recommendation.get("Assigned Pathway"))
    if pdf_specialization and pathway and pdf_specialization.casefold() != pathway.casefold():
        warnings.append(
            f"The transcript prints specialization '{pdf_specialization}'. "
            f"The Phase-1 study-plan match, built without {HOLD_OUT_LABEL}, is '{pathway}'. "
            "The recommendation uses the study-plan match."
        )
    if prerequisites.empty and not record["remaining"].empty:
        warnings.append(MSG_PREREQUISITE)
    engine_warning = _text(recommendation.get("Warning"))
    if engine_warning:
        warnings.append(engine_warning)

    blocked_statuses = {
        STATUS_BLOCKED_PREREQUISITE,
        STATUS_BLOCKED_MIXING,
        STATUS_CONDITIONAL,
        STATUS_MANUAL,
        STATUS_PATHWAY,
        NOT_ELIGIBLE,
    }
    selected_status = selected["Status"].astype(str) if not selected.empty and "Status" in selected.columns else pd.Series(dtype=str)
    violations = int(selected_status.isin(blocked_statuses).sum()) if not selected.empty else 0
    selected_codes = set(selected["Course"].astype(str)) if not selected.empty else set()
    blocked_codes = set(blocked["Course"].astype(str)) if not blocked.empty else set()
    blocked_selected = len(selected_codes & blocked_codes)
    credits = float(pd.to_numeric(selected["Credit Hours"], errors="coerce").fillna(0).sum()) if not selected.empty else 0.0
    load = record["load"]
    maximum_credits = _number(load.get("Maximum Credits"))
    maximum_courses = _number(load.get("Maximum Courses"))
    status = _text(recommendation.get("Recommendation Status"))
    within_load = True
    if maximum_credits is not None and credits > maximum_credits + 1e-6:
        within_load = False
    if maximum_courses is not None and len(selected) > maximum_courses:
        within_load = False
    load_label = "Valid" if status == STATUS_COMPLETE and within_load else "Requires Review"

    cards = _cards(profile, recommendation, record["summary"])
    stages = _stages(
        parsed,
        record,
        status=status,
        complete_status=STATUS_COMPLETE,
        has_scores=not selected.empty and selected["Rule Score"].notna().all(),
    )
    metadata = {
        "phase": "Phase 1 – Academic Rule-Based Recommendation",
        "student_code": _text(profile.get("Student Code")),
        "assigned_pathway": pathway,
        "pathway_readiness": _text(recommendation.get("Pathway Readiness")),
        "recommendation_status": status,
        "eligible_courses": int(len(eligible)),
        "recommended_courses": int(len(selected)),
        "recommended_credits": credits,
        "plan_is_subset_of_eligible": set(selected["Course"].astype(str)).issubset(set(eligible["Course"].astype(str))) if not selected.empty else True,
        "academic_load": load_label,
        "prerequisite_violations": violations,
        "blocked_courses_selected": blocked_selected,
        "maximum_courses": maximum_courses,
        "maximum_credits": maximum_credits,
        "holdout_policy": (
            f"{HOLD_OUT_LABEL} is the evaluation holdout. This demo does not read the holdout workbook "
            "and does not use Spring 2026 transcript rows to complete a course or to train a model."
        ),
        "scores": {"rule_score": "populated", **{name: None for name in FUTURE_SCORE_FIELDS}},
    }
    return RecommendationResult(
        student_profile=cards,
        course_history={
            "Completed": record["completed"],
            "Failed": record["failed"],
            "Withdrawn": record["withdrawn"],
            "Repeated": record["repeated"],
            "Remaining": record["remaining"],
        },
        remaining_courses=record["remaining"],
        prerequisite_results=prerequisites,
        candidate_courses=record["audit"],
        eligible_courses=eligible,
        selected_courses=selected,
        blocked_courses=blocked,
        warnings=list(dict.fromkeys(warnings)),
        metadata=metadata,
        stages=stages,
        extraction=parsed,
    )


def _present_selected(
    selected: pd.DataFrame,
    reasons: dict[str, str],
    holdout_codes: set[str],
) -> pd.DataFrame:
    columns = [
        "Rank", "Course", "Course Name", "Type", "Rule Score", "Status",
        "Recommendation Reason", "Credit Hours", "Priority Class", "Study Plan",
        "Remaining Requirement", "Holdout Note",
        "grade_prediction_score", "cf_score", "content_score", "transformer_score",
        "cgpa_impact_score", "hybrid_score",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in selected.iterrows():
        code = _text(row.get("Recommended Course Code"))
        reason = _text(row.get("Reason for Recommendation")) or reasons.get(_text(row.get("Advising Priority Class")), "")
        note = ""
        if code in holdout_codes:
            note = (
                f"Printed under {HOLD_OUT_LABEL} on this transcript. That term is held out, "
                "so the Phase-1 plan still treats the course from the pre-holdout record."
            )
        rows.append(
            {
                "Rank": row.get("Rank"),
                "Course": code,
                "Course Name": _text(row.get("Recommended Course Title")),
                "Type": _text(row.get("Course Type")),
                "Rule Score": pd.to_numeric(row.get("Rule-Based Score"), errors="coerce"),
                "Status": _text(row.get("Eligibility Status")),
                "Recommendation Reason": reason,
                "Credit Hours": pd.to_numeric(row.get("Credit Hours"), errors="coerce"),
                "Priority Class": _text(row.get("Advising Priority Class")),
                "Study Plan": _text(row.get("Study Plan Level")),
                "Remaining Requirement": _text(row.get("Remaining Reason")),
                "Holdout Note": note,
                "grade_prediction_score": pd.NA,
                "cf_score": pd.NA,
                "content_score": pd.NA,
                "transformer_score": pd.NA,
                "cgpa_impact_score": pd.NA,
                "hybrid_score": pd.NA,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(["Rank", "Course"], kind="mergesort").reset_index(drop=True)


def _present_eligible(
    audit: pd.DataFrame,
    selected: pd.DataFrame,
    reasons: dict[str, str],
    allowed_status: str,
) -> pd.DataFrame:
    """Rank every course the engine marked Allowed. Do not cut the list to a semester size."""

    columns = ["Priority", "Course", "Course Name", "Type", "Credits", "Rule Score", "Reason"]
    if audit.empty or "Candidate Status" not in audit.columns:
        return pd.DataFrame(columns=columns)
    allowed = audit.loc[audit["Candidate Status"].map(_text).eq(allowed_status)].copy()
    if allowed.empty:
        return pd.DataFrame(columns=columns)
    selected_reasons = {}
    if not selected.empty and "Course" in selected.columns:
        selected_reasons = {
            _text(row.get("Course")): _text(row.get("Recommendation Reason"))
            for _, row in selected.iterrows()
        }
    rows = []
    for _, row in allowed.iterrows():
        code = _text(row.get("Course Code"))
        priority_class = _text(row.get("Advising Priority Class"))
        rows.append(
            {
                "Priority": pd.to_numeric(row.get("Rule Priority"), errors="coerce"),
                "Course": code,
                "Course Name": _text(row.get("Course Title")),
                "Type": priority_class,
                "Credits": pd.to_numeric(row.get("Credit Hours"), errors="coerce"),
                "Rule Score": pd.to_numeric(row.get("Rule-Based Score"), errors="coerce"),
                "Reason": selected_reasons.get(code) or reasons.get(priority_class, ""),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(["Priority", "Course"], kind="mergesort").reset_index(drop=True)


def _present_blocked(audit: pd.DataFrame, allowed_status: str) -> pd.DataFrame:
    columns = ["Course", "Course Name", "Score", "Status", "Reason", "Priority Class"]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    work = audit.copy()
    status = work["Candidate Status"].map(_text) if "Candidate Status" in work.columns else pd.Series("", index=work.index)
    confirmed = status.eq(allowed_status)
    blocked = work.loc[~confirmed].copy()
    if blocked.empty:
        return pd.DataFrame(columns=columns)
    presented = pd.DataFrame(
        {
            "Course": blocked["Course Code"].map(_text),
            "Course Name": blocked["Course Title"].map(_text) if "Course Title" in blocked.columns else "",
            "Score": pd.to_numeric(blocked["Rule-Based Score"], errors="coerce") if "Rule-Based Score" in blocked.columns else pd.NA,
            "Status": blocked["Candidate Status"].map(_text),
            "Reason": blocked["Exclusion Reason"].map(_text) if "Exclusion Reason" in blocked.columns else "",
            "Priority Class": blocked["Advising Priority Class"].map(_text) if "Advising Priority Class" in blocked.columns else "",
        }
    )
    return presented.sort_values(["Status", "Course"], kind="mergesort").reset_index(drop=True)


def _cards(profile: dict[str, object], recommendation: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    cards: dict[str, object] = {}

    def put(label: str, value: object) -> None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        if str(value).strip() in {"", "nan", "None"}:
            return
        cards[label] = value

    put("Student Code", profile.get("Student Code"))
    put("Specialization", recommendation.get("Assigned Pathway") or profile.get("Current Specialization"))
    put("Current Study Level", profile.get("Current Level") or summary.get("Current Level"))
    put("Latest Semester", profile.get("Latest Semester"))
    put("CGPA", profile.get("Latest CGPA"))
    put("Completed Credits", profile.get("Total Credits Earned"))
    put("Remaining Courses", profile.get("Remaining Courses Count"))
    put("Failed Courses", profile.get("Failed Courses"))
    put("Withdrawn Courses", profile.get("Withdrawn Courses"))
    put("Pathway Readiness", recommendation.get("Pathway Readiness") or summary.get("Pathway Readiness"))
    put("Recommendation Status", recommendation.get("Recommendation Status"))
    return cards


def _stages(
    parsed: dict[str, object],
    record: dict[str, object],
    *,
    status: str,
    complete_status: str,
    has_scores: bool,
) -> list[dict[str, str]]:
    attempts = parsed.get("course_attempts")
    extracted = isinstance(attempts, pd.DataFrame) and not attempts.empty
    profile_ok = bool(_text(record["profile"].get("Student Code")))
    plan_ok = bool(_text(record["recommendation"].get("Assigned Pathway")))
    history_ok = all(isinstance(record[name], pd.DataFrame) for name in ("completed", "failed", "withdrawn"))
    remaining_ok = isinstance(record["remaining"], pd.DataFrame) and (
        not record["remaining"].empty or bool(record["summary"])
    )
    prerequisite_ok = isinstance(record["prerequisites"], pd.DataFrame) and not record["prerequisites"].empty
    eligibility_ok = isinstance(record["audit"], pd.DataFrame) and not record["audit"].empty
    priority_ok = has_scores or (status == complete_status and record["selected"].empty)
    plan_state = "confirmed" if status == complete_status else "review"
    return [
        _stage("Transcript extracted", "confirmed" if extracted else "failed"),
        _stage("Student academic profile created", "confirmed" if profile_ok else "failed"),
        _stage("Study plan identified", "confirmed" if plan_ok else "review"),
        _stage("Course attempts evaluated", "confirmed" if history_ok else "review"),
        _stage("Completed courses identified", "confirmed" if history_ok else "review"),
        _stage("Failed courses identified", "confirmed" if history_ok else "review"),
        _stage("Withdrawn courses identified", "confirmed" if history_ok else "review"),
        _stage("Remaining requirements calculated", "confirmed" if remaining_ok else "review"),
        _stage("Prerequisites checked", "confirmed" if prerequisite_ok else "review"),
        _stage("Eligibility rules applied", "confirmed" if eligibility_ok else "review"),
        _stage("Academic priority calculated", "confirmed" if priority_ok else "review"),
        _stage("Recommendation plan generated", plan_state),
    ]


def _stage(label: str, state: str) -> dict[str, str]:
    return {"label": label, "state": state}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
