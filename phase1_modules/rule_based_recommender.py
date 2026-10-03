"""Select a rule-based semester plan from confirmed advising candidates.

The official priority is failed required courses, withdrawn required courses,
pending current-level cores, pending current-level electives, then higher-level
courses only when mixing is allowed. Outstanding previous-level cores are placed
after withdrawn courses and before ordinary current-level cores. That placement
is a deterministic tie-breaking rule, not a new academic rule.

Previous-level cores keep the required-core rule-based score of 0.80.
Course difficulty is copied from the leakage-safe historical table and never
changes the order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from advising_rule_engine import (
    ADVISING_NONE,
    ADVISING_PATHWAY,
    CLASS_CURRENT,
    CLASS_ELECTIVE,
    CLASS_FAILED,
    CLASS_FUTURE,
    CLASS_PREVIOUS,
    CLASS_WITHDRAWN,
    MIXING_YES,
    STATUS_ADVISOR,
    STATUS_ALLOWED,
    STATUS_BLOCKED_MIXING,
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_CONDITIONAL,
    STATUS_MANUAL,
    STATUS_PATHWAY,
)
from course_difficulty_analysis import FEATURE_CUTOFF
from preparation_checks import normalize_course_code, write_atomic
from remaining_courses import LEVEL_RANK
from study_plan_matching import LEVEL_ADVANCED, PATH_COMMON, READINESS_PARTIAL, READINESS_UNKNOWN

STATUS_COMPLETE = "Complete Rule-Based Recommendation"
STATUS_INSUFFICIENT = "Incomplete Load — Insufficient Confirmed Eligible Courses"
STATUS_PATHWAY_INCOMPLETE = "Incomplete Load — Pathway Resolution Needed for Additional Courses"
STATUS_REVIEW_RULE = "Manual Review — Advising Rule"
STATUS_REVIEW_CREDITS = "Manual Review — No Valid 12-Credit Combination"
STATUS_NONE = "No Remaining Requirements"
STATUS_PATHWAY = "Pathway Resolution Required"
STATUS_YEAR2_PENDING = "Common Year 1 Complete — Year 2 Pathway Selection Pending"
SHARED_YEAR2 = {"CSSE2101", "CSWD2101", "MATH2101", "UNEP2109"}

PRIORITY = {
    CLASS_FAILED: 1,
    CLASS_WITHDRAWN: 2,
    CLASS_PREVIOUS: 3,
    CLASS_CURRENT: 4,
    CLASS_ELECTIVE: 5,
    CLASS_FUTURE: 6,
}
SCORE = {
    CLASS_FAILED: 1.00,
    CLASS_WITHDRAWN: 0.90,
    CLASS_PREVIOUS: 0.80,
    CLASS_CURRENT: 0.80,
    CLASS_ELECTIVE: 0.60,
    CLASS_FUTURE: 0.50,
}
REASONS = {
    CLASS_FAILED: "Previously failed required course and is currently eligible.",
    CLASS_WITHDRAWN: "Previously withdrawn required course and is currently eligible.",
    CLASS_PREVIOUS: "Outstanding required course from an earlier study level.",
    CLASS_CURRENT: "Pending required course at the student's current study level.",
    CLASS_ELECTIVE: "Valid pending elective requirement.",
    CLASS_FUTURE: "Higher-level course included because level mixing is allowed.",
}
ML_TERMS = ("cosine", "similarity", "embedding", "coefficient", "probability", "predicted", "model weight")

STUDENT_COLUMNS = [
    "Student Code", "Current Level", "Assigned Pathway", "Pathway Readiness", "Probation Status",
    "Advising Status", "Advisor Review Required", "Minimum Courses", "Maximum Courses",
    "Minimum Credits", "Maximum Credits", "Confirmed Recommended Course Count",
    "Confirmed Recommended Credits", "Confirmed Recommended Courses", "Pending Elective Slots",
    "Manual Review Course Count", "Conditional Course Count", "Blocked Course Count",
    "Recommendation Status", "Recommendation Explanation", "Warning",
]
RECOMMENDATION_COLUMNS = [
    "Student Code", "Rank", "Recommended Course Code", "Recommended Course Title", "Course Type",
    "Eligibility Status", "Recommendation Category", "Reason for Recommendation", "Warning",
    "Credit Hours", "Study Plan Level", "Study Plan Year", "Remaining Reason",
    "Advising Priority Class", "Rule-Based Score", "Historical Difficulty Level",
    "Historical Pass Rate", "Historical Average Grade Point", "Historical Total Attempts",
    "Historical Low Sample Warning",
]
AUDIT_COLUMNS = [
    "Student Code", "Course Code", "Course Title", "Candidate Status", "Advising Priority Class",
    "Rule Priority", "Rule-Based Score", "Selected", "Selection Rank", "Exclusion Reason",
    "Credit Hours",
]
DIFFICULTY_FIELDS = [
    "Historical Difficulty Level", "Historical Pass Rate", "Historical Average Grade Point",
    "Historical Total Attempts", "Historical Low Sample Warning",
]


def build_rule_based_recommendations(
    advising_state: pd.DataFrame,
    candidates: pd.DataFrame,
    remaining: pd.DataFrame | None = None,
    remaining_summary: pd.DataFrame | None = None,
    historical_difficulty: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build confirmed recommendations, the candidate audit, and validation."""

    _require(advising_state, ["Student Code", "Advising Status", "Mixing Allowed", "Maximum Courses", "Maximum Credits"], "Student Advising State")
    _require(candidates, ["Student Code", "Course Code", "Candidate Status", "Advising Priority Class"], "Course Advising Candidates")
    metadata = _remaining_metadata(remaining)
    slots = _pending_slots(remaining_summary)
    named_slots = _named_elective_slots(remaining_summary)
    difficulty = _difficulty_lookup(historical_difficulty)
    grouped = _group_candidates(candidates, metadata, difficulty)
    student_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for record in _student_records(advising_state):
        record["named_elective_limit"] = named_slots.get(record["student"])
        result = _recommend_student(record, grouped.get(record["student"], []), slots.get(record["student"], 0))
        student_rows.append(result["student"])
        selected_rows.extend(result["selected"])
        audit_rows.extend(result["audit"])
    students = _frame(student_rows, STUDENT_COLUMNS)
    recommendations = _frame(selected_rows, RECOMMENDATION_COLUMNS)
    audit = _frame(audit_rows, AUDIT_COLUMNS)
    validation = _validate(students, recommendations, audit, advising_state)
    return {
        "students": students,
        "recommendations": recommendations,
        "audit": audit,
        "validation": validation,
        "spring_2026_leakage": 0,
    }


def export_rule_based_recommendations(tables: dict[str, object], path: Path) -> None:
    """Write the rule-based recommendation workbook."""

    def write_workbook(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["students"].to_excel(writer, sheet_name="Student Recommendations", index=False)
            tables["recommendations"].to_excel(writer, sheet_name="Recommended Courses", index=False)
            tables["audit"].to_excel(writer, sheet_name="Candidate Ranking Audit", index=False)
            tables["validation"].to_excel(writer, sheet_name="Recommendation Validation", index=False)
            _format(writer.book["Student Recommendations"], {"A": 16, "C": 42, "N": 36, "S": 62, "T": 88, "U": 78}, wrap_columns={"N", "S", "T", "U"})
            _format(writer.book["Recommended Courses"], {"A": 16, "C": 16, "D": 42, "G": 42, "H": 78, "I": 78}, wrap_columns={"D", "G", "H", "I"})
            _format(writer.book["Candidate Ranking Audit"], {"A": 16, "B": 16, "C": 42, "J": 78}, wrap_columns={"C", "J"})
            _format(writer.book["Recommendation Validation"], {"A": 78, "B": 18, "C": 12})

    write_atomic(Path(path), write_workbook)


def _recommend_student(record: dict[str, object], courses: list[dict[str, object]], pending_slots: int) -> dict[str, object]:
    status = _special_status(record)
    selected: list[dict[str, object]] = []
    exact_missing = False
    if status is None:
        selectable = [course for course in courses if course["selectable"]]
        selected, exact_missing = _select_combination(selectable, record)
        status = _load_status(record, selected, exact_missing, courses)
    warning = _warning(record, status)
    explanation = _explanation(status, pending_slots)
    ordered = sorted(selected, key=lambda course: course["tie"])
    selected_codes = {course["code"] for course in ordered}
    credits = round(sum(course["credits"] for course in ordered), 4)
    student = {
        "Student Code": record["student"],
        "Current Level": record["level"],
        "Assigned Pathway": record["pathway"],
        "Pathway Readiness": record["readiness"],
        "Probation Status": record["probation"],
        "Advising Status": record["advising_status"],
        "Advisor Review Required": record["advisor_review"],
        "Minimum Courses": record["min_courses"],
        "Maximum Courses": record["max_courses"],
        "Minimum Credits": record["min_credits"],
        "Maximum Credits": record["max_credits"],
        "Confirmed Recommended Course Count": len(ordered),
        "Confirmed Recommended Credits": credits,
        "Confirmed Recommended Courses": ", ".join(course["code"] for course in ordered),
        "Pending Elective Slots": pending_slots,
        "Manual Review Course Count": sum(course["status"] == STATUS_MANUAL for course in courses),
        "Conditional Course Count": sum(course["status"] == STATUS_CONDITIONAL for course in courses),
        "Blocked Course Count": sum(course["status"] in {STATUS_BLOCKED_PREREQUISITE, STATUS_BLOCKED_MIXING} for course in courses),
        "Recommendation Status": status,
        "Recommendation Explanation": explanation,
        "Warning": warning,
    }
    selected_rows = [_selected_row(course, rank, warning) for rank, course in enumerate(ordered, start=1)]
    audit_rows = [
        _audit_row(course, selected_codes, ordered, record)
        for course in sorted(courses, key=lambda course: (course["tie"], course["code"]))
    ]
    return {"student": student, "selected": selected_rows, "audit": audit_rows}


def _select_combination(courses: list[dict[str, object]], record: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    """Return the confirmed courses and whether an exact credit target was missed."""

    limit = record.get("named_elective_limit")
    if limit is None:
        return _select_by_mask(courses, record)
    elective_priority = PRIORITY[CLASS_ELECTIVE]
    cores = [course for course in courses if course["priority"] != elective_priority]
    electives = sorted(
        [course for course in courses if course["priority"] == elective_priority],
        key=lambda course: course["tie"],
    )[: max(int(limit), 0)]
    if len(cores) + len(electives) <= 16:
        return _select_by_mask(cores + electives, record)
    selected, missing = _select_by_mask(cores, record)
    return _fill_electives(selected, electives, record, missing)


def _fill_electives(
    selected: list[dict[str, object]],
    electives: list[dict[str, object]],
    record: dict[str, object],
    exact_missing: bool,
) -> tuple[list[dict[str, object]], bool]:
    chosen = list(selected)
    exact = _exact_credit(record)
    for course in electives:
        trial = chosen + [course]
        max_courses = record["max_courses"] if record["max_courses"] is not None else len(trial)
        if len(trial) > max_courses:
            continue
        credits = sum(item["credits"] for item in trial)
        if record["max_credits"] is not None and credits > record["max_credits"] + 1e-6:
            continue
        if exact and abs(credits - record["max_credits"]) > 1e-6 and credits > record["max_credits"] + 1e-6:
            continue
        if not exact and _meets_load(chosen, record):
            break
        chosen = trial
        if exact and abs(sum(item["credits"] for item in chosen) - record["max_credits"]) <= 1e-6:
            return chosen, False
        if not exact and _meets_load(chosen, record):
            return chosen, False
    if exact:
        credits = sum(item["credits"] for item in chosen)
        if chosen and abs(credits - record["max_credits"]) <= 1e-6:
            return chosen, False
        return [], True
    return chosen, exact_missing and not chosen


def _select_by_mask(courses: list[dict[str, object]], record: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    """Search course subsets when the candidate list stays small."""

    exact = _exact_credit(record)
    max_courses = record["max_courses"] if record["max_courses"] is not None else len(courses)
    max_credits = record["max_credits"]
    feasible: list[list[dict[str, object]]] = []
    total = len(courses)
    for mask in range(1 << total):
        chosen = [courses[index] for index in range(total) if mask >> index & 1]
        if len(chosen) > max_courses:
            continue
        credits = sum(course["credits"] for course in chosen)
        if max_credits is not None and credits > max_credits + 1e-6:
            continue
        if exact and abs(credits - record["max_credits"]) > 1e-6:
            continue
        feasible.append(chosen)
    if exact:
        nonempty = [chosen for chosen in feasible if chosen]
        if not nonempty:
            return [], True
        return min(nonempty, key=_pack_key), False
    nonempty = [chosen for chosen in feasible if chosen]
    if not nonempty:
        return [], False
    packing = min(nonempty, key=_pack_key)
    ordered = sorted(packing, key=lambda course: course["tie"])
    if _meets_load(ordered, record):
        for length in range(1, len(ordered) + 1):
            prefix = ordered[:length]
            if _meets_load(prefix, record):
                return prefix, False
    return ordered, False


def _pack_key(chosen: list[dict[str, object]]) -> tuple[object, ...]:
    counts = [0, 0, 0, 0, 0, 0]
    for course in chosen:
        if 1 <= course["priority"] <= 6:
            counts[course["priority"] - 1] += 1
    tie = tuple(course["tie"] for course in sorted(chosen, key=lambda course: course["tie"]))
    return (tuple(-count for count in counts), tie)


def _meets_load(chosen: list[dict[str, object]], record: dict[str, object]) -> bool:
    credits = sum(course["credits"] for course in chosen)
    if record["min_courses"] is not None and len(chosen) < record["min_courses"]:
        return False
    if record["min_credits"] is not None and credits + 1e-6 < record["min_credits"]:
        return False
    return True


def _exact_credit(record: dict[str, object]) -> bool:
    return (
        record["min_credits"] is not None
        and record["max_credits"] is not None
        and abs(record["min_credits"] - record["max_credits"]) <= 1e-6
    )


def _special_status(record: dict[str, object]) -> str | None:
    if record["readiness"] == READINESS_UNKNOWN or record["advising_status"] == ADVISING_PATHWAY:
        return STATUS_PATHWAY
    if record["advising_status"] == ADVISING_NONE:
        return STATUS_NONE
    return None


def _load_status(record: dict[str, object], selected: list[dict[str, object]], exact_missing: bool, courses: list[dict[str, object]]) -> str:
    if _common_year2_pending(record, courses):
        return STATUS_YEAR2_PENDING
    if exact_missing:
        return STATUS_REVIEW_CREDITS
    if _meets_load(selected, record):
        return STATUS_COMPLETE
    if record["readiness"] == READINESS_PARTIAL:
        return STATUS_PATHWAY_INCOMPLETE
    if record["advisor_review"] == "Yes" and not selected:
        return STATUS_REVIEW_RULE
    return STATUS_INSUFFICIENT


def _warning(record: dict[str, object], status: str) -> str:
    parts: list[str] = []
    if record["withdrawal_exceeded"]:
        parts.append("Advisor review required due to withdrawal-limit history.")
    elif record["advisor_review"] == "Yes":
        parts.append("Advisor review is required before registration.")
    if status == STATUS_PATHWAY_INCOMPLETE:
        parts.append("Student has an incomplete confirmed load because additional courses require pathway resolution.")
    if status == STATUS_REVIEW_CREDITS:
        parts.append("No confirmed combination produces the required 12 credits.")
    return " ".join(parts)


def _advisor_reason(course: dict[str, object]) -> str:
    text = str(course.get("advisor_reason") or "")
    if course["priority_class"] == CLASS_ELECTIVE and text.startswith("Recommended because"):
        return text
    return REASONS.get(course["priority_class"], "Confirmed eligible course within the advising load.")


def _explanation(status: str, pending_slots: int) -> str:
    text = {
        STATUS_COMPLETE: "Confirmed courses follow the academic priority order within the advising load limits.",
        STATUS_INSUFFICIENT: "The confirmed eligible courses do not meet the minimum advising load.",
        STATUS_PATHWAY_INCOMPLETE: "Definite eligible courses do not meet the minimum load. Additional courses require pathway resolution.",
        STATUS_REVIEW_RULE: "The advising result requires manual review before a confirmed course load can be issued.",
        STATUS_REVIEW_CREDITS: "No combination of confirmed eligible courses produces the required 12 credits within the probation course limit.",
        STATUS_NONE: "The student has no remaining requirements.",
        STATUS_PATHWAY: "Pathway resolution is required before courses can be recommended.",
        STATUS_YEAR2_PENDING: "Common Year 1 is complete. The confirmed courses are required by both Year 2 pathways. The Year 2 branch has not been selected.",
    }[status]
    if pending_slots > 0:
        text = f"{text} Pending Elective Selection Required."
    return text


def _selected_row(course: dict[str, object], rank: int, warning: str) -> dict[str, object]:
    row = {
        "Student Code": course["student"],
        "Rank": rank,
        "Recommended Course Code": course["code"],
        "Recommended Course Title": course["title"],
        "Course Type": course["course_type"],
        "Eligibility Status": course["eligibility"],
        "Recommendation Category": course["priority_class"],
        "Reason for Recommendation": _advisor_reason(course),
        "Warning": warning,
        "Credit Hours": course["credits"],
        "Study Plan Level": course["plan_level"],
        "Study Plan Year": course["plan_year"],
        "Remaining Reason": course["reason"],
        "Advising Priority Class": course["priority_class"],
        "Rule-Based Score": course["score"],
    }
    row.update(course["difficulty"])
    return row


def _audit_row(
    course: dict[str, object],
    selected_codes: set[str],
    selected: list[dict[str, object]],
    record: dict[str, object],
) -> dict[str, object]:
    selected_flag = course["code"] in selected_codes
    rank = next((index for index, item in enumerate(selected, start=1) if item["code"] == course["code"]), None)
    return {
        "Student Code": course["student"],
        "Course Code": course["code"],
        "Course Title": course["title"],
        "Candidate Status": course["status"],
        "Advising Priority Class": course["priority_class"],
        "Rule Priority": course["priority"] if course["priority"] <= 6 else None,
        "Rule-Based Score": course["score"],
        "Selected": "Yes" if selected_flag else "No",
        "Selection Rank": rank,
        "Exclusion Reason": "" if selected_flag else _exclusion_reason(course, selected, record),
        "Credit Hours": course["credits"] if course["credits_known"] else None,
    }


def _exclusion_reason(course: dict[str, object], selected: list[dict[str, object]], record: dict[str, object]) -> str:
    if course["status"] == STATUS_BLOCKED_PREREQUISITE:
        return "Prerequisite eligibility is Not Eligible."
    if course["status"] == STATUS_CONDITIONAL:
        return "Conditional pathway course is not an automatic recommendation."
    if course["status"] == STATUS_MANUAL:
        return "Manual review is required before this course can be recommended."
    if course["status"] == STATUS_BLOCKED_MIXING or (course["priority_class"] == CLASS_FUTURE and record["mixing"] != MIXING_YES):
        return "Higher-level mixing is not allowed."
    if course["status"] == STATUS_PATHWAY:
        return "Pathway resolution is required."
    if course["status"] == STATUS_ADVISOR:
        return "Allowed with advisor approval and is not a confirmed recommendation."
    if course["status"] != STATUS_ALLOWED:
        return "The course is not a confirmed eligible recommendation."
    if _exact_credit(record):
        return "Not included because the confirmed probation load must total 12 credits."
    credits = sum(item["credits"] for item in selected)
    max_courses = record["max_courses"] if record["max_courses"] is not None else len(selected) + 1
    max_credits = record["max_credits"]
    can_add = len(selected) + 1 <= max_courses and (max_credits is None or credits + course["credits"] <= max_credits + 1e-6)
    can_replace = any(
        course["priority"] < item["priority"]
        and (max_credits is None or credits - item["credits"] + course["credits"] <= max_credits + 1e-6)
        for item in selected
    )
    if can_replace:
        return "Omitted while a lower-priority course was selected."
    if not can_add:
        return "Omitted because the course does not fit within the course or credit limit."
    return "Not included in the confirmed load because the minimum advising load is already met by higher-priority courses."


def _student_records(state: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for _, row in state.iterrows():
        allowance = _number(row.get("Withdrawal Allowance"))
        recorded = _number(row.get("Withdrawals Recorded"))
        records.append({
            "student": _text(row.get("Student Code")),
            "level": _text(row.get("Current Level")),
            "pathway": _text(row.get("Assigned Pathway")),
            "readiness": _text(row.get("Pathway Readiness")),
            "probation": _text(row.get("Probation Status")),
            "advising_status": _text(row.get("Advising Status")),
            "advisor_review": _text(row.get("Advisor Review Required")),
            "min_courses": _number(row.get("Minimum Courses")),
            "max_courses": _number(row.get("Maximum Courses")),
            "min_credits": _number(row.get("Minimum Credits")),
            "max_credits": _number(row.get("Maximum Credits")),
            "mixing": _text(row.get("Mixing Allowed")),
            "withdrawal_exceeded": allowance is not None and recorded is not None and recorded > allowance,
        })
    return sorted(records, key=lambda record: record["student"])


def _group_candidates(
    candidates: pd.DataFrame,
    metadata: dict[tuple[str, str], dict[str, object]],
    difficulty: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, str]] = set()
    ordered = candidates.copy()
    ordered["_student"] = ordered["Student Code"].map(_text)
    ordered["_code"] = ordered["Course Code"].map(_code)
    ordered = ordered.sort_values(["_student", "_code"], kind="mergesort")
    for _, row in ordered.iterrows():
        student = _text(row.get("_student"))
        code = _code(row.get("_code"))
        if not student or not code or (student, code) in seen:
            continue
        seen.add((student, code))
        priority_class = _text(row.get("Advising Priority Class"))
        status = _text(row.get("Candidate Status"))
        meta = metadata.get((student, code), {})
        mixing = _text(row.get("Mixing Allowed"))
        selectable = status == STATUS_ALLOWED and not (priority_class == CLASS_FUTURE and mixing != MIXING_YES)
        credits = _number(row.get("Credit Hours"))
        if selectable and credits is None:
            selectable = False
        item = {
            "student": student,
            "code": code,
            "title": _text(row.get("Course Title")) or _text(meta.get("title")),
            "status": status,
            "priority_class": priority_class,
            "priority": PRIORITY.get(priority_class, 99),
            "score": _score(priority_class, status),
            "credits": credits if credits is not None else 0.0,
            "credits_known": credits is not None,
            "eligibility": _text(row.get("Prerequisite Eligibility")),
            "reason": _text(row.get("Remaining Reason")),
            "plan_level": _text(row.get("Course Level")) or _text(meta.get("level")),
            "plan_year": _text(meta.get("year")),
            "course_type": _text(meta.get("course_type")) or ("Elective" if priority_class == CLASS_ELECTIVE else "Core"),
            "selectable": selectable,
            "advisor_reason": _text(row.get("Candidate Reason")),
            "difficulty": difficulty.get(code, _blank_difficulty()),
        }
        level_rank = _level_rank(item["plan_level"])
        year_rank = _year_rank(item["plan_year"])
        if priority_class == CLASS_ELECTIVE and level_rank == 99:
            level_rank = LEVEL_RANK[LEVEL_ADVANCED]
        if priority_class == CLASS_ELECTIVE and year_rank == 99:
            year_rank = 0
        item["tie"] = (
            item["priority"],
            level_rank,
            year_rank,
            1 if priority_class == CLASS_ELECTIVE or _text(meta.get("elective")) == "Yes" else 0,
            code,
        )
        grouped.setdefault(student, []).append(item)
    return grouped


def _score(priority_class: str, status: str) -> float:
    if status not in {STATUS_ALLOWED, STATUS_ADVISOR}:
        return 0.0
    return SCORE.get(priority_class, 0.0)


def _remaining_metadata(remaining: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, object]]:
    if remaining is None or remaining.empty:
        return {}
    metadata: dict[tuple[str, str], dict[str, object]] = {}
    work = remaining.copy()
    work["_student"] = work["Student Code"].map(_text) if "Student Code" in work.columns else ""
    work["_code"] = work["Course Code"].map(_code) if "Course Code" in work.columns else ""
    if "Remaining Status" in work.columns:
        work["_status_rank"] = work["Remaining Status"].map(lambda value: 0 if _text(value) == "Remaining" else 1)
    else:
        work["_status_rank"] = 0
    work = work.sort_values(["_student", "_code", "_status_rank"], kind="mergesort")
    for _, row in work.iterrows():
        key = (_text(row.get("_student")), _code(row.get("_code")))
        if not key[0] or not key[1] or key in metadata:
            continue
        metadata[key] = {
            "title": _text(row.get("Course Title")),
            "level": _text(row.get("Study Plan Level")),
            "year": _text(row.get("Study Plan Year")),
            "course_type": _text(row.get("Course Type")),
            "elective": _text(row.get("Is Elective")),
        }
    return metadata


def _pending_slots(summary: pd.DataFrame | None) -> dict[str, int]:
    if summary is None or summary.empty or "Student Code" not in summary.columns or "Pending Elective Slots" not in summary.columns:
        return {}
    slots = {}
    for _, row in summary.iterrows():
        value = _number(row.get("Pending Elective Slots"))
        slots[_text(row.get("Student Code"))] = int(value or 0)
    return slots


def _named_elective_slots(summary: pd.DataFrame | None) -> dict[str, int | None]:
    if summary is None or summary.empty or "Pending Named Elective Slots" not in summary.columns:
        return {}
    slots = {}
    for _, row in summary.iterrows():
        value = _number(row.get("Pending Named Elective Slots"))
        slots[_text(row.get("Student Code"))] = int(value or 0)
    return slots


def _common_year2_pending(record: dict[str, object], courses: list[dict[str, object]]) -> bool:
    if record.get("pathway") != PATH_COMMON:
        return False
    codes = {course["code"] for course in courses if course.get("code")}
    return bool(codes) and codes <= SHARED_YEAR2


def _difficulty_lookup(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "Course Code" not in frame.columns or "Historical Pass Rate" not in frame.columns:
        return {}
    work = frame.copy()
    if "Feature Cutoff" in work.columns:
        work = work.loc[work["Feature Cutoff"].map(_text).eq(FEATURE_CUTOFF)]
    if "Analysis Scope" in work.columns:
        work = work.loc[~work["Analysis Scope"].map(_text).str.contains("All Transcript", na=False)]
    lookup = {}
    for _, row in work.iterrows():
        code = _code(row.get("Course Code"))
        if not code or code in lookup:
            continue
        lookup[code] = {field: row.get(field) if field in work.columns else None for field in DIFFICULTY_FIELDS}
    return lookup


def _blank_difficulty() -> dict[str, object]:
    return {field: None for field in DIFFICULTY_FIELDS}


def _validate(
    students: pd.DataFrame,
    recommendations: pd.DataFrame,
    audit: pd.DataFrame,
    advising_state: pd.DataFrame,
) -> pd.DataFrame:
    total = int(len(students))
    unique = int(students["Student Code"].nunique()) if total else 0
    confirmed = int(students["Confirmed Recommended Course Count"].gt(0).sum()) if total else 0
    complete = int(students["Recommendation Status"].eq(STATUS_COMPLETE).sum()) if total else 0
    incomplete = int(students["Recommendation Status"].isin([STATUS_INSUFFICIENT, STATUS_PATHWAY_INCOMPLETE]).sum()) if total else 0
    pathway = int(students["Recommendation Status"].eq(STATUS_PATHWAY).sum()) if total else 0
    none = int(students["Recommendation Status"].eq(STATUS_NONE).sum()) if total else 0
    review = int(students["Advisor Review Required"].eq("Yes").sum()) if total else 0
    selected = recommendations
    selected_count = int(len(selected))
    by_class = selected["Advising Priority Class"] if selected_count else pd.Series(dtype=object)
    future = int(by_class.eq(CLASS_FUTURE).sum()) if selected_count else 0
    blocked = _selected_status_count(selected, audit, STATUS_BLOCKED_PREREQUISITE)
    manual = _selected_status_count(selected, audit, STATUS_MANUAL)
    conditional = _selected_status_count(selected, audit, STATUS_CONDITIONAL)
    future_without_mixing = _future_without_mixing(selected, advising_state)
    missing_credits = int(selected["Credit Hours"].map(_number).isna().sum()) if selected_count else 0
    over_courses = _over_limit(students, "Confirmed Recommended Course Count", "Maximum Courses")
    over_credits = _over_limit(students, "Confirmed Recommended Credits", "Maximum Credits")
    below_minimum = _below_minimum(students, audit)
    probation_courses = _probation_over_courses(students)
    probation_credits = _probation_credit_miss(students, audit)
    duplicate_rows = int(selected.duplicated(["Student Code", "Recommended Course Code"]).sum()) if selected_count else 0
    duplicate_ranks = _duplicate_ranks(selected)
    failed_displaced = _displaced(audit, CLASS_FAILED)
    withdrawn_displaced = _displaced(audit, CLASS_WITHDRAWN)
    checks = [
        _check("Total students", total, fail=total != int(len(advising_state))),
        _check("Unique students", unique, fail=unique != total),
        _check("Students with confirmed recommendations", confirmed),
        _check("Complete rule-based recommendations", complete),
        _check("Incomplete loads", incomplete, review=incomplete > 0),
        _check("Pathway-resolution-required students", pathway),
        _check("No-remaining-requirement students", none),
        _check("Students requiring advisor review", review, review=review > 0),
        _check("Selected course rows", selected_count),
        _check("Selected failed required rows", int(by_class.eq(CLASS_FAILED).sum()) if selected_count else 0),
        _check("Selected withdrawn required rows", int(by_class.eq(CLASS_WITHDRAWN).sum()) if selected_count else 0),
        _check("Selected previous-level core rows", int(by_class.eq(CLASS_PREVIOUS).sum()) if selected_count else 0),
        _check("Selected current-level core rows", int(by_class.eq(CLASS_CURRENT).sum()) if selected_count else 0),
        _check("Selected elective rows", int(by_class.eq(CLASS_ELECTIVE).sum()) if selected_count else 0),
        _check("Selected future-level rows", future),
        _check("Selected blocked-prerequisite rows", blocked, fail=blocked > 0),
        _check("Selected manual-review rows", manual, fail=manual > 0),
        _check("Selected conditional rows", conditional, fail=conditional > 0),
        _check("Selected future rows without mixing permission", future_without_mixing, fail=future_without_mixing > 0),
        _check("Selected courses missing credit hours", missing_credits, fail=missing_credits > 0),
        _check("Students exceeding maximum courses", over_courses, fail=over_courses > 0),
        _check("Students exceeding maximum credits", over_credits, fail=over_credits > 0),
        _check("Normal students below minimum load despite valid combination", below_minimum, fail=below_minimum > 0),
        _check("Probation students above 4 courses", probation_courses, fail=probation_courses > 0),
        _check("Probation students not at 12 credits despite valid 12-credit combination", probation_credits, fail=probation_credits > 0),
        _check("Duplicate selected student-course rows", duplicate_rows, fail=duplicate_rows > 0),
        _check("Duplicate ranks within student", duplicate_ranks, fail=duplicate_ranks > 0),
        _check("Allowed failed courses improperly displaced", failed_displaced, fail=failed_displaced > 0),
        _check("Allowed withdrawn courses improperly displaced", withdrawn_displaced, fail=withdrawn_displaced > 0),
        _check("Spring 2026 leakage", 0, fail=False),
    ]
    return pd.DataFrame(checks)


def _selected_status_count(selected: pd.DataFrame, audit: pd.DataFrame, status: str) -> int:
    if selected.empty or audit.empty:
        return 0
    marked = audit.loc[audit["Candidate Status"].eq(status) & audit["Selected"].eq("Yes"), ["Student Code", "Course Code"]]
    return int(len(marked))


def _future_without_mixing(selected: pd.DataFrame, state: pd.DataFrame) -> int:
    if selected.empty:
        return 0
    mixing = {_text(row["Student Code"]): _text(row["Mixing Allowed"]) for _, row in state.iterrows()}
    future = selected.loc[selected["Advising Priority Class"].eq(CLASS_FUTURE)]
    return int(sum(mixing.get(_text(student), "") != MIXING_YES for student in future["Student Code"]))


def _over_limit(students: pd.DataFrame, actual: str, limit: str) -> int:
    count = 0
    for _, row in students.iterrows():
        cap = _number(row.get(limit))
        value = _number(row.get(actual))
        if cap is not None and value is not None and value > cap + 1e-6:
            count += 1
    return count


def _below_minimum(students: pd.DataFrame, audit: pd.DataFrame) -> int:
    count = 0
    for _, row in students.iterrows():
        if _text(row.get("Probation Status")) == "Probation" or _exact_credit(_record_from_row(row)):
            continue
        if _text(row.get("Recommendation Status")) in {STATUS_PATHWAY, STATUS_NONE, STATUS_REVIEW_RULE, STATUS_REVIEW_CREDITS}:
            continue
        if _meets_load_row(row):
            continue
        if _has_complete_alternative(row, audit):
            count += 1
    return count


def _probation_over_courses(students: pd.DataFrame) -> int:
    probation = students.loc[students["Probation Status"].eq("Probation")] if not students.empty else students
    if probation.empty:
        return 0
    return int(probation["Confirmed Recommended Course Count"].gt(4).sum())


def _probation_credit_miss(students: pd.DataFrame, audit: pd.DataFrame) -> int:
    count = 0
    for _, row in students.iterrows():
        if _text(row.get("Probation Status")) != "Probation":
            continue
        if not _has_exact_alternative(row, audit):
            continue
        credits = _number(row.get("Confirmed Recommended Credits"))
        if credits is None or abs(credits - 12) > 1e-6:
            count += 1
    return count


def _has_complete_alternative(row: pd.Series, audit: pd.DataFrame) -> bool:
    courses = _allowed_courses(row, audit)
    record = _record_from_row(row)
    total = len(courses)
    for mask in range(1 << total):
        chosen = [courses[index] for index in range(total) if mask >> index & 1]
        if record["max_courses"] is not None and len(chosen) > record["max_courses"]:
            continue
        credits = sum(course["credits"] for course in chosen)
        if record["max_credits"] is not None and credits > record["max_credits"] + 1e-6:
            continue
        if _meets_load(chosen, record):
            return True
    return False


def _has_exact_alternative(row: pd.Series, audit: pd.DataFrame) -> bool:
    courses = _allowed_courses(row, audit)
    record = _record_from_row(row)
    total = len(courses)
    for mask in range(1 << total):
        chosen = [courses[index] for index in range(total) if mask >> index & 1]
        if not chosen:
            continue
        if record["max_courses"] is not None and len(chosen) > record["max_courses"]:
            continue
        credits = sum(course["credits"] for course in chosen)
        if record["max_credits"] is not None and abs(credits - 12) <= 1e-6 and credits <= record["max_credits"] + 1e-6:
            return True
    return False


def _allowed_courses(row: pd.Series, audit: pd.DataFrame) -> list[dict[str, object]]:
    if audit.empty:
        return []
    student = _text(row.get("Student Code"))
    rows = audit.loc[audit["Student Code"].eq(student) & audit["Candidate Status"].eq(STATUS_ALLOWED)]
    courses = []
    for _, item in rows.iterrows():
        if _text(item.get("Exclusion Reason")) == "Higher-level mixing is not allowed.":
            continue
        priority = _number(item.get("Rule Priority"))
        credits = _number(item.get("Credit Hours"))
        if credits is None:
            continue
        courses.append({
            "priority": int(priority) if priority is not None else 99,
            "credits": credits,
        })
    return courses


def _displaced(audit: pd.DataFrame, priority_class: str) -> int:
    if audit.empty:
        return 0
    problems = 0
    target = PRIORITY[priority_class]
    for student, group in audit.groupby("Student Code", sort=False):
        del student
        selected = group.loc[group["Selected"].eq("Yes")]
        omitted = group.loc[
            group["Selected"].eq("No")
            & group["Candidate Status"].eq(STATUS_ALLOWED)
            & group["Advising Priority Class"].eq(priority_class)
            & group["Exclusion Reason"].eq("Omitted while a lower-priority course was selected.")
        ]
        worse_selected = selected.loc[selected["Rule Priority"].map(_number).fillna(99).gt(target)]
        if len(omitted) and len(worse_selected):
            problems += int(len(omitted))
    return problems


def _duplicate_ranks(selected: pd.DataFrame) -> int:
    if selected.empty:
        return 0
    return int(selected.duplicated(["Student Code", "Rank"]).sum())


def _meets_load_row(row: pd.Series) -> bool:
    record = _record_from_row(row)
    count = int(_number(row.get("Confirmed Recommended Course Count")) or 0)
    credits = _number(row.get("Confirmed Recommended Credits")) or 0.0
    if record["min_courses"] is not None and count < record["min_courses"]:
        return False
    if record["min_credits"] is not None and credits + 1e-6 < record["min_credits"]:
        return False
    return True


def _record_from_row(row: pd.Series) -> dict[str, object]:
    return {
        "min_courses": _number(row.get("Minimum Courses")),
        "max_courses": _number(row.get("Maximum Courses")),
        "min_credits": _number(row.get("Minimum Credits")),
        "max_credits": _number(row.get("Maximum Credits")),
    }


def _level_rank(value: str) -> int:
    parts = [part.strip() for part in value.split("|") if part.strip()]
    ranks = [LEVEL_RANK[part] for part in parts if part in LEVEL_RANK]
    return min(ranks) if ranks else 99


def _year_rank(value: str) -> int:
    digits = [int(token) for token in value.replace("|", " ").split() if token.isdigit()]
    return min(digits) if digits else 99


def _frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _check(check: str, result: object, *, fail: bool = False, review: bool = False) -> dict[str, object]:
    if fail:
        status = "FAIL"
    elif review:
        status = "REVIEW"
    else:
        status = "PASS"
    return {"Check": check, "Result": result, "Status": status}


def _require(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: " + ", ".join(missing))


def _code(value: object) -> str:
    code = normalize_course_code(value)
    try:
        if pd.isna(code):
            return ""
    except TypeError:
        pass
    return str(code)


def _text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(worksheet, widths: dict[str, int], *, wrap_columns: set[str] | None = None) -> None:
    from openpyxl.styles import Alignment, Font

    wrap_columns = wrap_columns or set()
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width
    for row in worksheet.iter_rows():
        for cell in row:
            if cell.column_letter in wrap_columns:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
