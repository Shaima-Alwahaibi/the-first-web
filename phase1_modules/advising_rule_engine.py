"""Apply official advising rules to remaining courses.

The engine reads the project Advising Rules sheet and the current profile,
mapping, remaining-course, and prerequisite outputs. It records the permitted
load, priority class, mixing decision, and advisor-review state.

It does not select a final set of courses and it does not calculate a CGPA.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from preparation_checks import normalize_course_code, write_atomic
from repeat_classification import (
    REPEAT_FAILURE as DERIVED_FAILURE,
    REPEAT_IMPROVEMENT as DERIVED_IMPROVEMENT,
    REPEAT_UNRESOLVED as DERIVED_UNRESOLVED,
    REPEAT_WITHDRAWAL as DERIVED_WITHDRAWAL,
    derive_repeat_classification,
)
from prerequisite_checker import (
    ELIGIBLE,
    ELIGIBLE_ADVISOR,
    EVIDENCE_INSUFFICIENT,
    NOT_ELIGIBLE,
    PATHWAY_REQUIRED,
)
from remaining_courses import (
    LEVEL_RANK,
    REASON_FAILED,
    REASON_WITHDRAWN,
    SCOPE_COMMON,
    SCOPE_CURRENT,
    SCOPE_FUTURE,
    SCOPE_SPECIFIC,
    PLACEMENT_MISSING,
    STATUS_CONDITIONAL as REMAINING_CONDITIONAL,
    STATUS_PATHWAY_REQUIRED,
    STATUS_REMAINING,
)
from student_profiles import GRADE_RANK, normalize_grade
from study_plan_matching import HOLDOUT_TOKEN, LEVEL_ADVANCED, LEVEL_BACHELOR, LEVEL_DIPLOMA, PATH_DSAI, READINESS_UNKNOWN

MIXING_YES = "Yes"
MIXING_NO = "No"
MIXING_REVIEW = "Manual Review"
MIXING_NOT_APPLICABLE = "Not applicable"
MIXING_NOT_EVALUATED = "Not evaluated"

PROGRESSION_SATISFIED = "satisfied"
PROGRESSION_NOT_SATISFIED = "not_satisfied"
PROGRESSION_UNAVAILABLE = "unavailable"

STATUS_ALLOWED = "Allowed"
STATUS_BLOCKED_PREREQUISITE = "Blocked by Prerequisite"
STATUS_ADVISOR = "Allowed with Advisor Approval"
STATUS_CONDITIONAL = "Conditional Pathway"
STATUS_MANUAL = "Manual Review"
STATUS_PATHWAY = "Pathway Resolution Required"
STATUS_BLOCKED_MIXING = "Blocked by Level Mixing"

CLASS_FAILED = "Failed Required"
CLASS_WITHDRAWN = "Withdrawn Required"
CLASS_CURRENT = "Pending Current-Level Core"
CLASS_PREVIOUS = "Pending Previous-Level Core"
CLASS_ELECTIVE = "Pending Current-Level Elective Requirement"
CLASS_FUTURE = "Future/Higher-Level Course"
CLASS_CONDITIONAL = "Conditional Specialization Course"
CLASS_MANUAL = "Manual Review"

LEVEL_CURRENT = "Current Level"
LEVEL_PREVIOUS = "Previous Level"
LEVEL_FUTURE = "Future Level"
LEVEL_UNRESOLVED = "Unresolved"

ADVISING_PATHWAY = "Pathway Resolution Required"
ADVISING_NONE = "No Remaining Requirements"
ADVISING_REVIEW = "Manual Review Required"
ADVISING_READY = "Envelope Determined"

REPEAT_FAILED = "Repeated because previously failed"
REPEAT_WITHDRAWAL = "Repeated after withdrawal"
REPEAT_IMPROVE = "Repeated to improve GPA"
REPEAT_UNRESOLVED = "Repeat type unresolved"
REVIEW_YES = "Yes"
REVIEW_NO = "No"

CATEGORY_BY_AREA = {
    "Course Load": "Course Load",
    "Probation": "Probation Load",
    "Failed Core Courses": "Failed Course Priority",
    "Withdrawn Courses": "Withdrawn Course Priority",
    "Withdrawal Limit": "Other",
    "Mixing Levels": "Level Mixing",
    "Mixing Load": "Level Mixing",
    "DSAI Criteria": "Specialization Entry",
    "Foundation Courses": "Other",
}

STATE_COLUMNS = [
    "Student Code",
    "Current Level",
    "Assigned Pathway",
    "Pathway Readiness",
    "Possible Pathways",
    "Probation Status",
    "Academic Risk",
    "Remaining Current-Level Courses",
    "Remaining Previous-Level Courses",
    "Remaining Future-Level Courses",
    "Failed Required Courses",
    "Withdrawn Required Courses",
    "Eligible Remaining Courses",
    "Not Eligible Remaining Courses",
    "Advisor-Approval Eligibility Count",
    "Manual Review Prerequisite Count",
    "Minimum Courses",
    "Maximum Courses",
    "Minimum Credits",
    "Maximum Credits",
    "Mixing Allowed",
    "Mixing Reason",
    "Repeat Present",
    "Repeat Type",
    "Mixing + Repeat Conflict",
    "Advisor Review Required",
    "Advising Status",
    "Withdrawal Allowance",
    "Withdrawals Recorded",
    "Specialization Entry Evidence",
    "Load Rule",
]
CANDIDATE_COLUMNS = [
    "Student Code",
    "Course Code",
    "Course Title",
    "Credit Hours",
    "Current Level",
    "Course Level",
    "Requirement Scope",
    "Remaining Reason",
    "Prerequisite Eligibility",
    "Eligibility Evidence Status",
    "Conditional Pathway",
    "Advising Priority Class",
    "Current-Level Status",
    "Mixing Required",
    "Mixing Allowed",
    "Repeat Interaction",
    "Advisor Approval Required",
    "Candidate Status",
    "Candidate Reason",
]
AUDIT_COLUMNS = [
    "Rule Area",
    "Official Rule",
    "Source/Notes",
    "Internal Rule Category",
    "Implemented",
    "Implementation Method",
    "Needs Manual Review",
    "Notes",
]
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


@dataclass(frozen=True)
class GradeCondition:
    minimum: str
    title: str


@dataclass(frozen=True)
class ParsedAdvisingRules:
    normal_min_courses: int | None
    normal_max_courses: int | None
    normal_min_credits: int | None
    normal_max_credits: int | None
    probation_max_courses: int | None
    probation_credits: int | None
    probation_credits_exact: bool
    mixing_threshold: int | None
    mixing_max_courses: int | None
    mixing_lcgpa: float | None
    mixing_extra_courses: int | None
    progression: str
    withdrawal_limits: dict[str, int]
    grade_conditions: tuple[GradeCondition, ...]
    conditions_required: int | None
    cgpa_minimum: float | None
    foundation_prefix: str
    audit: pd.DataFrame


def interpret_advising_rules(rules: pd.DataFrame) -> ParsedAdvisingRules:
    """Classify every official advising row and read the limits it states."""

    _require_columns(rules, ["Rule Area", "Rule", "Source/Notes"], "Advising Rules")
    indexed = {_text(row["Rule Area"]): row for _, row in rules.iterrows()}
    audit_rows: list[dict[str, object]] = []

    def add(area: str, method: str, *, implemented: str = REVIEW_YES, review: str = REVIEW_NO, notes: str = "") -> None:
        source = indexed.get(area)
        audit_rows.append({
            "Rule Area": area,
            "Official Rule": _text(source["Rule"]) if source is not None else "",
            "Source/Notes": _text(source["Source/Notes"]) if source is not None else "",
            "Internal Rule Category": CATEGORY_BY_AREA.get(area, "Other"),
            "Implemented": implemented if source is not None else REVIEW_NO,
            "Implementation Method": method if source is not None else "The official sheet does not contain this rule area.",
            "Needs Manual Review": review if source is not None else REVIEW_YES,
            "Notes": notes,
        })

    load = _parse_course_load(indexed.get("Course Load"))
    add(
        "Course Load",
        "Sets the non-probation envelope from the stated course and credit ranges.",
        implemented=REVIEW_YES if load else REVIEW_NO,
        review=REVIEW_NO if load else REVIEW_YES,
        notes="Source/Notes mentions an average of 15 credits. That average is not used as a minimum or a maximum.",
    )
    probation = _parse_probation(indexed.get("Probation"))
    probation_note = "The rule states a maximum of 4 courses / 12 credits."
    if probation and probation[2]:
        probation_note = "Source/Notes says probation registration is only 12 credit hours, so the credit envelope is 12 to 12. A minimum course count is not stated."
    add(
        "Probation",
        "Caps probation students at the stated course maximum and credit limit. The stored profile probation flag is used unchanged.",
        implemented=REVIEW_YES if probation else REVIEW_NO,
        review=REVIEW_NO if probation else REVIEW_YES,
        notes=probation_note,
    )
    add(
        "Failed Core Courses",
        "Marks Remaining Reason Failed Required Course as the highest advising priority class. Prerequisite blocks remain in force.",
    )
    add(
        "Withdrawn Courses",
        "Marks Remaining Reason Withdrawn Required Course so the requirement can be considered again. Prerequisite blocks remain in force.",
    )
    limits = _parse_withdrawal_limits(indexed.get("Withdrawal Limit"))
    add(
        "Withdrawal Limit",
        "Compares historical withdrawals in each official level with the stated per-level allowance.",
        implemented=REVIEW_YES if limits else REVIEW_NO,
        review=REVIEW_NO if limits else REVIEW_YES,
        notes="The allowance limits further withdrawals. It does not remove a withdrawn requirement from consideration.",
    )
    threshold = _parse_mixing_threshold(indexed.get("Mixing Levels"))
    add(
        "Mixing Levels",
        "Allows higher-level mixing only when current-level courses are within the stated count and progression evidence is satisfied.",
        implemented=REVIEW_YES if threshold else REVIEW_NO,
        review=REVIEW_YES,
        notes=(
            "The rule requires LCGPA and English criteria, and the sheet does not state the LCGPA threshold or the English course and grade. "
            "Repeat and mixing are not granted together; that combination stays in manual review and requires advisor approval."
        ),
    )
    mixing_load = _parse_mixing_load(indexed.get("Mixing Load"))
    add(
        "Mixing Load",
        "When mixing is allowed, replaces the normal course maximum with the stated mixing load. LCGPA of 3 or above can add one higher-level course.",
        implemented=REVIEW_YES if mixing_load else REVIEW_NO,
        review=REVIEW_NO if mixing_load else REVIEW_YES,
        notes="The mixing-load maximum does not raise the probation maximum.",
    )
    dsai = _parse_dsai(indexed.get("DSAI Criteria"))
    add(
        "DSAI Criteria",
        "Evaluates the stated any-two grade and CGPA conditions as evidence. The result does not assign a pathway.",
        implemented=REVIEW_YES if dsai else REVIEW_NO,
        review=REVIEW_NO if dsai else REVIEW_YES,
        notes="Calculus I and Mathematics for Computing are identified from Study Plan Courses titles.",
    )
    prefix = _parse_foundation_prefix(indexed.get("Foundation Courses"))
    add(
        "Foundation Courses",
        "Keeps course codes with the stated prefix out of selectable advising candidates.",
        implemented=REVIEW_YES if prefix else REVIEW_NO,
        review=REVIEW_NO if prefix else REVIEW_YES,
        notes="A foundation code may still appear in a prerequisite explanation. It is not a recommendation candidate.",
    )
    known = set(CATEGORY_BY_AREA)
    for area, source in indexed.items():
        if area in known:
            continue
        audit_rows.append({
            "Rule Area": area,
            "Official Rule": _text(source["Rule"]),
            "Source/Notes": _text(source["Source/Notes"]),
            "Internal Rule Category": "Other",
            "Implemented": REVIEW_NO,
            "Implementation Method": "The source area has no implemented advising action.",
            "Needs Manual Review": REVIEW_YES,
            "Notes": "This official row is preserved and is not interpreted as an extra numeric rule.",
        })
    audit = _frame(audit_rows, AUDIT_COLUMNS, ["Rule Area"])
    return ParsedAdvisingRules(
        normal_min_courses=load[0] if load else None,
        normal_max_courses=load[1] if load else None,
        normal_min_credits=load[2] if load else None,
        normal_max_credits=load[3] if load else None,
        probation_max_courses=probation[0] if probation else None,
        probation_credits=probation[1] if probation else None,
        probation_credits_exact=bool(probation[2]) if probation else False,
        mixing_threshold=threshold,
        mixing_max_courses=mixing_load[0] if mixing_load else None,
        mixing_lcgpa=mixing_load[1] if mixing_load else None,
        mixing_extra_courses=mixing_load[2] if mixing_load else None,
        progression=PROGRESSION_UNAVAILABLE,
        withdrawal_limits=limits,
        grade_conditions=dsai[0] if dsai else (),
        conditions_required=dsai[1] if dsai else None,
        cgpa_minimum=dsai[2] if dsai else None,
        foundation_prefix=prefix or "FP",
        audit=audit,
    )


def course_load_envelope(
    parsed: ParsedAdvisingRules,
    *,
    probation: bool,
    mixing_allowed: str,
    lcgpa: float | None,
) -> dict[str, object]:
    """Return the permitted course and credit bounds for one student."""

    if probation:
        minimum_credits = parsed.probation_credits if parsed.probation_credits_exact else None
        return {
            "Minimum Courses": None,
            "Maximum Courses": parsed.probation_max_courses,
            "Minimum Credits": minimum_credits,
            "Maximum Credits": parsed.probation_credits,
            "Load Rule": (
            f"Probation students can take maximum {parsed.probation_max_courses} courses / {parsed.probation_credits} credits. "
            + ("Source/Notes allows only 12 credit hours." if parsed.probation_credits_exact else "A minimum credit value is not stated.")
        ),
        }
    maximum = parsed.normal_max_courses
    load_rule = "Normal students can take 4 to 6 courses, equivalent to 12 to 18 credits per semester."
    if mixing_allowed == MIXING_YES and parsed.mixing_max_courses is not None:
        maximum = parsed.mixing_max_courses
        load_rule = "Mixing load is normally 4 courses, within 12 to 18 credits."
        if parsed.mixing_lcgpa is not None and lcgpa is not None and lcgpa >= parsed.mixing_lcgpa and parsed.mixing_extra_courses is not None:
            maximum = parsed.mixing_extra_courses
            load_rule = "Mixing load is 5 courses because LCGPA is 3 or above, by adding one higher-level course, within 12 to 18 credits."
        elif parsed.mixing_lcgpa is not None and lcgpa is None:
            load_rule = "Mixing load stays at the stated normal mixing maximum because LCGPA is unavailable for the higher exception."
    return {
        "Minimum Courses": parsed.normal_min_courses,
        "Maximum Courses": maximum,
        "Minimum Credits": parsed.normal_min_credits,
        "Maximum Credits": parsed.normal_max_credits,
        "Load Rule": load_rule,
    }


def decide_mixing(
    *,
    current_courses: int,
    current_courses_upper: int,
    future_courses: int,
    conditional_future_courses: int,
    unnamed_future_slots: int,
    specialization_current_courses: int,
    threshold: int,
    progression: str,
    active_repeat: bool,
) -> dict[str, object]:
    """Decide whether higher-level mixing is allowed for one student."""

    if future_courses <= 0 and conditional_future_courses <= 0:
        return _mixing(MIXING_NOT_APPLICABLE, "Not applicable — no higher-level courses remain", conflict=False, advisor=False)
    if current_courses > threshold:
        return _mixing(MIXING_NO, f"No — {current_courses} current-level courses remain", conflict=False, advisor=False)
    if current_courses_upper > threshold:
        return _mixing(
            MIXING_REVIEW,
            "Manual Review — current-level course count is unresolved because pending elective slots are not split into current and previous levels",
            conflict=active_repeat,
            advisor=active_repeat,
        )
    if current_courses <= threshold and specialization_current_courses > 0:
        return _mixing(
            MIXING_REVIEW,
            "Manual Review — mixing depends on specialization-specific remaining requirements",
            conflict=active_repeat,
            advisor=active_repeat,
        )
    if future_courses <= 0 and conditional_future_courses > 0:
        return _mixing(
            MIXING_REVIEW,
            "Manual Review — higher-level courses depend on the specialization pathway",
            conflict=active_repeat,
            advisor=active_repeat,
        )
    if future_courses <= 0 and unnamed_future_slots > 0:
        return _mixing(
            MIXING_REVIEW,
            "Manual Review — higher-level requirements remain as elective slots and no course has been selected",
            conflict=active_repeat,
            advisor=active_repeat,
        )
    if progression == PROGRESSION_NOT_SATISFIED:
        return _mixing(MIXING_NO, "No — progression requirements are not satisfied", conflict=False, advisor=False)
    if progression == PROGRESSION_UNAVAILABLE:
        reason = "Manual Review — progression evidence unresolved"
        if active_repeat:
            reason += "; course repetition and higher-level mixing require advisor approval"
        return _mixing(MIXING_REVIEW, reason, conflict=active_repeat, advisor=active_repeat)
    if active_repeat:
        return _mixing(
            MIXING_REVIEW,
            "Manual Review — course repetition and higher-level mixing require advisor approval",
            conflict=True,
            advisor=True,
        )
    return _mixing(
        MIXING_YES,
        f"Yes — {current_courses} current-level courses remain and progression requirements satisfied",
        conflict=False,
        advisor=False,
    )


def build_advising_output(
    profiles: pd.DataFrame,
    remaining: pd.DataFrame,
    prerequisites: pd.DataFrame,
    study_plan: pd.DataFrame,
    advising_rules: pd.DataFrame,
    history: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
    resolution: pd.DataFrame | None = None,
    remaining_summary: pd.DataFrame | None = None,
    electives: pd.DataFrame | None = None,
    progression_status: str | None = None,
) -> dict[str, object]:
    """Build the advising state, candidate classes, rule audit, and validation."""

    parsed = interpret_advising_rules(advising_rules)
    if parsed.mixing_threshold is None or parsed.normal_min_courses is None or parsed.probation_max_courses is None:
        raise ValueError("Advising Rules does not state the course-load, probation, and mixing limits.")
    progression = parsed.progression if progression_status is None else progression_status
    historical, leakage = _historical_rows(history)
    credits = _credit_catalog(study_plan)
    titles = _title_catalog(study_plan)
    _guard_previous_outputs(profiles, mapping, resolution, remaining_summary)
    summary = _student_directory(profiles, remaining, remaining_summary, mapping)
    prerequisite_index = _prerequisite_index(prerequisites)
    repeats = _repeat_index(historical, profiles, remaining)
    withdrawals = _withdrawal_index(historical, parsed.withdrawal_limits)
    specializations = _specialization_index(historical, summary, titles, parsed)
    remaining_groups = _group_rows(remaining, "Student Code")
    elective_groups = _group_rows(electives, "Student Code")

    states: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for record in summary:
        student_remaining = remaining_groups.get(record["student"], [])
        student_electives = elective_groups.get(record["student"], [])
        built = _advise_student(
            record,
            student_remaining,
            student_electives,
            prerequisite_index,
            repeats.get(record["student"], _empty_repeat()),
            withdrawals.get(record["student"], _empty_withdrawal(parsed.withdrawal_limits)),
            specializations.get(record["student"], "Not applicable. The possible pathways do not include Data Science and Artificial Intelligence."),
            credits,
            parsed,
            progression,
        )
        states.append(built["state"])
        candidates.extend(built["candidates"])

    state_frame = _frame(states, STATE_COLUMNS, ["Student Code"])
    candidate_frame = _frame(candidates, CANDIDATE_COLUMNS, ["Student Code", "Course Code", "Conditional Pathway"])
    validation = _validate(state_frame, candidate_frame, profiles, parsed, leakage, prerequisite_index)
    return {
        "state": state_frame,
        "candidates": candidate_frame,
        "audit": parsed.audit,
        "validation": validation,
        "spring_2026_leakage": leakage,
        "parsed": parsed,
    }


def export_advising_results(tables: dict[str, object], path) -> None:
    """Write the advising workbook without changing earlier outputs."""

    def write_workbook(target) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["state"].to_excel(writer, sheet_name="Student Advising State", index=False)
            tables["candidates"].to_excel(writer, sheet_name="Course Advising Candidates", index=False)
            tables["audit"].to_excel(writer, sheet_name="Advising Rule Audit", index=False)
            tables["validation"].to_excel(writer, sheet_name="Advising Validation", index=False)
            _format(writer.book["Student Advising State"], {"A": 16, "C": 42, "D": 26, "U": 28, "V": 78, "AE": 78}, wrap_columns={"U", "V", "AD", "AE"})
            _format(writer.book["Course Advising Candidates"], {"A": 16, "B": 16, "C": 42, "T": 72}, wrap_columns={"C", "T"})
            _format(writer.book["Advising Rule Audit"], {"A": 28, "B": 78, "C": 78, "F": 78, "H": 78}, wrap_columns={"B", "C", "F", "H"})
            _format(writer.book["Advising Validation"], {"A": 78, "B": 18, "C": 12})

    write_atomic(path, write_workbook)


def _advise_student(
    record: dict[str, object],
    rows: list[dict[str, object]],
    elective_slots: list[dict[str, object]],
    prerequisites: dict[tuple[str, str, str], dict[str, object]],
    repeat: dict[str, object],
    withdrawal: dict[str, object],
    specialization: str,
    credits: dict[str, float | None],
    parsed: ParsedAdvisingRules,
    progression: str,
) -> dict[str, object]:
    unknown = record["readiness"] == READINESS_UNKNOWN or record["readiness"] == ADVISING_PATHWAY
    classified: list[dict[str, object]] = []
    for row in rows:
        code = _code(row.get("Course Code"))
        if code == "":
            continue
        level_status = _level_status(_text(row.get("Current Level")), _text(row.get("Study Plan Level")), _text(row.get("Requirement Scope")), _text(row.get("Remaining Status")))
        prerequisite = prerequisites.get((record["student"], code, _text(row.get("Conditional Pathway"))), {})
        classified.append({"row": row, "code": code, "level_status": level_status, "prerequisite": prerequisite})

    definite = [item for item in classified if _text(item["row"].get("Remaining Status")) == STATUS_REMAINING]
    current_courses = sum(item["level_status"] == LEVEL_CURRENT for item in definite)
    previous_courses = sum(item["level_status"] == LEVEL_PREVIOUS for item in definite)
    future_courses = sum(item["level_status"] == LEVEL_FUTURE for item in definite)
    conditional_future = sum(
        _text(item["row"].get("Remaining Status")) == REMAINING_CONDITIONAL and item["level_status"] == LEVEL_FUTURE
        for item in classified
    )
    specialization_current = sum(
        _text(item["row"].get("Remaining Status")) == REMAINING_CONDITIONAL and item["level_status"] == LEVEL_CURRENT
        for item in classified
    )
    ambiguous_slots, future_slots, conditional_future_slots, specialization_slots = _slot_counts(elective_slots)
    specialization_current += specialization_slots
    if future_courses <= 0:
        conditional_future += conditional_future_slots
    failed = sum(_text(item["row"].get("Remaining Reason")) == REASON_FAILED and _text(item["row"].get("Remaining Status")) == STATUS_REMAINING for item in classified)
    withdrawn = sum(_text(item["row"].get("Remaining Reason")) == REASON_WITHDRAWN and _text(item["row"].get("Remaining Status")) == STATUS_REMAINING for item in classified)
    has_remaining = bool(classified) or ambiguous_slots or future_slots or conditional_future_slots or specialization_slots
    active_repeat = bool(repeat["active"]) or failed > 0

    if unknown:
        mixing = _mixing(MIXING_NOT_EVALUATED, "Not evaluated — pathway resolution is required", conflict=False, advisor=False)
        envelope = {"Minimum Courses": None, "Maximum Courses": None, "Minimum Credits": None, "Maximum Credits": None, "Load Rule": ""}
    elif not has_remaining:
        mixing = _mixing(MIXING_NOT_APPLICABLE, "Not applicable — no remaining requirements", conflict=False, advisor=False)
        envelope = course_load_envelope(parsed, probation=record["probation"], mixing_allowed=mixing["allowed"], lcgpa=record["lcgpa"])
    else:
        mixing = decide_mixing(
            current_courses=current_courses,
            current_courses_upper=current_courses + ambiguous_slots,
            future_courses=future_courses,
            conditional_future_courses=conditional_future,
            unnamed_future_slots=future_slots if future_courses <= 0 and conditional_future <= 0 else 0,
            specialization_current_courses=specialization_current,
            threshold=parsed.mixing_threshold or 0,
            progression=progression,
            active_repeat=active_repeat,
        )
        envelope = course_load_envelope(parsed, probation=record["probation"], mixing_allowed=mixing["allowed"], lcgpa=record["lcgpa"])

    candidate_rows: list[dict[str, object]] = []
    if not unknown:
        for item in classified:
            candidate_rows.append(_candidate_row(record, item, credits, parsed.foundation_prefix, mixing, repeat, active_repeat))

    allowed = sum(row["Candidate Status"] == STATUS_ALLOWED for row in candidate_rows)
    blocked = sum(row["Candidate Status"] == STATUS_BLOCKED_PREREQUISITE for row in candidate_rows)
    advisor_eligibility = sum(row["Prerequisite Eligibility"] == ELIGIBLE_ADVISOR for row in candidate_rows)
    manual_prereq = sum(_manual_prerequisite(row) for row in candidate_rows)
    review_rows = [
        row for row in candidate_rows
        if row["Candidate Reason"] != "No prerequisite rule is listed for this course, so it is not confirmed."
    ]
    advisor_review = mixing["advisor"] or withdrawal["exceeded"] or any(row["Advisor Approval Required"] == REVIEW_YES for row in review_rows) or any(row["Candidate Status"] == STATUS_MANUAL for row in review_rows)
    if unknown:
        advising_status = ADVISING_PATHWAY
    elif not has_remaining:
        advising_status = ADVISING_NONE
    elif advisor_review or mixing["allowed"] == MIXING_REVIEW:
        advising_status = ADVISING_REVIEW
    else:
        advising_status = ADVISING_READY
    state = {
        "Student Code": record["student"],
        "Current Level": record["level"],
        "Assigned Pathway": record["pathway"],
        "Pathway Readiness": record["readiness"],
        "Possible Pathways": record["possible"],
        "Probation Status": record["probation_status"],
        "Academic Risk": record["risk"],
        "Remaining Current-Level Courses": current_courses,
        "Remaining Previous-Level Courses": previous_courses,
        "Remaining Future-Level Courses": future_courses,
        "Failed Required Courses": failed,
        "Withdrawn Required Courses": withdrawn,
        "Eligible Remaining Courses": allowed,
        "Not Eligible Remaining Courses": blocked,
        "Advisor-Approval Eligibility Count": advisor_eligibility,
        "Manual Review Prerequisite Count": manual_prereq,
        "Minimum Courses": envelope["Minimum Courses"],
        "Maximum Courses": envelope["Maximum Courses"],
        "Minimum Credits": envelope["Minimum Credits"],
        "Maximum Credits": envelope["Maximum Credits"],
        "Mixing Allowed": mixing["allowed"],
        "Mixing Reason": mixing["reason"],
        "Repeat Present": repeat["present"],
        "Repeat Type": repeat["type"],
        "Mixing + Repeat Conflict": REVIEW_YES if mixing["conflict"] else REVIEW_NO,
        "Advisor Review Required": REVIEW_YES if advisor_review else REVIEW_NO,
        "Advising Status": advising_status,
        "Withdrawal Allowance": withdrawal["allowance"],
        "Withdrawals Recorded": withdrawal["recorded"],
        "Specialization Entry Evidence": specialization,
        "Load Rule": envelope["Load Rule"],
    }
    return {"state": state, "candidates": candidate_rows}


def _candidate_row(
    record: dict[str, object],
    item: dict[str, object],
    credits: dict[str, float | None],
    foundation_prefix: str,
    mixing: dict[str, object],
    repeat: dict[str, object],
    active_repeat: bool,
) -> dict[str, object]:
    row = item["row"]
    code = item["code"]
    prerequisite = item["prerequisite"]
    eligibility = _text(prerequisite.get("Eligibility Status"))
    evidence = _text(prerequisite.get("Eligibility Evidence Status"))
    manual = _text(prerequisite.get("Manual Review Required"))
    remaining_status = _text(row.get("Remaining Status"))
    reason = _text(row.get("Remaining Reason"))
    level_status = item["level_status"]
    credit = credits.get(code)
    if credit is None:
        credit = _number(row.get("Credit Hours"))
    priority = _priority_class(remaining_status, reason, level_status, row)
    mixing_required = REVIEW_YES if level_status == LEVEL_FUTURE and remaining_status == STATUS_REMAINING else REVIEW_NO
    if _text(prerequisite.get("Eligibility Reason")) == "Missing source rule":
        status, candidate_reason, advisor = (
            STATUS_MANUAL,
            "No prerequisite rule is listed for this course, so it is not confirmed.",
            False,
        )
    else:
        status, candidate_reason, advisor = _candidate_decision(
            code,
            foundation_prefix,
            remaining_status,
            eligibility,
            evidence,
            manual,
            level_status,
            mixing,
            active_repeat,
            bool(prerequisite),
            _text(row.get("Study Plan Level")),
            _text(prerequisite.get("Candidate Scope")),
        )
    if status == STATUS_ALLOWED and _flag(row.get("Is Elective")):
        candidate_reason = _elective_advisor_reason(row, prerequisite)
    return {
        "Student Code": record["student"],
        "Course Code": code,
        "Course Title": _text(row.get("Course Title")),
        "Credit Hours": credit,
        "Current Level": record["level"],
        "Course Level": _text(row.get("Study Plan Level")),
        "Requirement Scope": _text(row.get("Requirement Scope")),
        "Remaining Reason": reason,
        "Prerequisite Eligibility": eligibility,
        "Eligibility Evidence Status": evidence,
        "Conditional Pathway": _text(row.get("Conditional Pathway")),
        "Advising Priority Class": priority,
        "Current-Level Status": level_status,
        "Mixing Required": mixing_required,
        "Mixing Allowed": mixing["allowed"],
        "Repeat Interaction": _repeat_interaction(code, reason, level_status, repeat, active_repeat, mixing_required),
        "Advisor Approval Required": REVIEW_YES if advisor else REVIEW_NO,
        "Candidate Status": status,
        "Candidate Reason": candidate_reason,
    }


def _candidate_decision(
    code: str,
    foundation_prefix: str,
    remaining_status: str,
    eligibility: str,
    evidence: str,
    manual: str,
    level_status: str,
    mixing: dict[str, object],
    active_repeat: bool,
    has_prerequisite: bool,
    plan_level: str = "",
    candidate_scope: str = "",
) -> tuple[str, str, bool]:
    del plan_level
    if code.startswith(foundation_prefix):
        return STATUS_MANUAL, "Foundation course codes are excluded from advising candidates and remain prerequisite references.", False
    if candidate_scope == "Conditional SE Candidate":
        return STATUS_CONDITIONAL, "Conditional SE candidate. The course is valid under Software Engineering only, so it is not a confirmed recommendation while the pathway is unresolved.", False
    if candidate_scope == "Conditional DSAI Candidate":
        return STATUS_CONDITIONAL, "Conditional DSAI candidate. The course is valid under Data Science and Artificial Intelligence only, so it is not a confirmed recommendation while the pathway is unresolved.", False
    if candidate_scope == "Pathway-Dependent Eligibility":
        return STATUS_CONDITIONAL, "Pathway-dependent eligibility. The pathway rules do not give the same result, so the course is not a confirmed recommendation.", False
    if remaining_status == REMAINING_CONDITIONAL:
        return STATUS_CONDITIONAL, "Conditional specialization course. It is not a definite course-load candidate.", False
    if not has_prerequisite:
        return STATUS_MANUAL, "Prerequisite result is missing for this remaining course.", True
    if evidence == EVIDENCE_INSUFFICIENT or manual == REVIEW_YES:
        return STATUS_MANUAL, "The prerequisite is required by the project reference Excel, but the available transcript data does not contain student-level evidence for that foundation course." if evidence == EVIDENCE_INSUFFICIENT else "Prerequisite evidence requires manual review.", True
    if eligibility == NOT_ELIGIBLE:
        return STATUS_BLOCKED_PREREQUISITE, "Prerequisite eligibility is Not Eligible.", False
    if eligibility == PATHWAY_REQUIRED:
        return STATUS_PATHWAY, "Pathway resolution is required.", False
    advisor = eligibility == ELIGIBLE_ADVISOR
    if level_status == LEVEL_FUTURE:
        if mixing["allowed"] == MIXING_NO:
            return STATUS_BLOCKED_MIXING, mixing["reason"], False
        if mixing["allowed"] == MIXING_REVIEW:
            return STATUS_MANUAL, mixing["reason"], bool(mixing["advisor"] or advisor or active_repeat)
        if active_repeat:
            return STATUS_ADVISOR, "Higher-level mixing is combined with course repetition and requires advisor approval.", True
        if advisor:
            return STATUS_ADVISOR, "Prerequisite eligibility requires advisor approval.", True
        return STATUS_ALLOWED, "Higher-level mixing is allowed and the prerequisite is satisfied.", False
    if advisor:
        return STATUS_ADVISOR, "Prerequisite eligibility requires advisor approval.", True
    if eligibility == ELIGIBLE:
        return STATUS_ALLOWED, "The course is inside the current advising scope and the prerequisite is satisfied.", False
    return STATUS_MANUAL, "Prerequisite eligibility is not an official selectable status.", True


def _elective_advisor_reason(row: dict[str, object], prerequisite: dict[str, object]) -> str:
    pool = _text(row.get("Elective Pool")) or "major elective"
    satisfied = _text(prerequisite.get("Satisfied Prerequisites"))
    if satisfied:
        return (
            f"Recommended because this is a remaining {pool}, its prerequisite {satisfied} has been completed, "
            "and the course fits the student's allowed load."
        )
    return (
        f"Recommended because this is a remaining {pool}, its prerequisites are satisfied, "
        "and the course fits the student's allowed load."
    )


def _priority_class(remaining_status: str, reason: str, level_status: str, row: dict[str, object]) -> str:
    if remaining_status == REMAINING_CONDITIONAL or _text(row.get("Requirement Scope")) == SCOPE_SPECIFIC:
        return CLASS_CONDITIONAL
    if reason == REASON_FAILED:
        return CLASS_FAILED
    if reason == REASON_WITHDRAWN:
        return CLASS_WITHDRAWN
    if _flag(row.get("Is Elective")):
        return CLASS_ELECTIVE
    if level_status == LEVEL_FUTURE:
        return CLASS_FUTURE
    if level_status == LEVEL_PREVIOUS:
        return CLASS_PREVIOUS
    if level_status == LEVEL_CURRENT:
        return CLASS_CURRENT
    return CLASS_MANUAL


def _repeat_interaction(code: str, reason: str, level_status: str, repeat: dict[str, object], active_repeat: bool, mixing_required: str) -> str:
    if reason == REASON_FAILED:
        return "Active course repetition"
    if active_repeat and mixing_required == REVIEW_YES:
        return "Repeat and higher-level mixing require advisor approval"
    if code in repeat["improvement"]:
        return "Completed repeat for grade improvement does not reopen the requirement"
    if level_status == LEVEL_FUTURE:
        return "None"
    return "None"


def _validate(
    state: pd.DataFrame,
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
    parsed: ParsedAdvisingRules,
    leakage: int,
    prerequisites: dict[tuple[str, str, str], dict[str, object]],
) -> pd.DataFrame:
    total = int(len(state))
    unique = int(state["Student Code"].nunique()) if total else 0
    profile_students = int(profiles["Student Code"].nunique()) if "Student Code" in profiles.columns else 0
    probation = state["Probation Status"].eq("Probation")
    unknown = state["Pathway Readiness"].eq(READINESS_UNKNOWN)
    normal = ~probation
    probation_limit = int((probation & state["Maximum Courses"].eq(parsed.probation_max_courses)).sum())
    normal_range = int((normal & ~unknown & state["Minimum Courses"].eq(parsed.normal_min_courses) & state["Maximum Courses"].eq(parsed.normal_max_courses)).sum())
    mixing_yes = int(state["Mixing Allowed"].eq(MIXING_YES).sum())
    mixing_no = int(state["Mixing Allowed"].eq(MIXING_NO).sum())
    mixing_review = int(state["Mixing Allowed"].eq(MIXING_REVIEW).sum())
    failed_rows = int(candidates["Advising Priority Class"].eq(CLASS_FAILED).sum()) if not candidates.empty else 0
    withdrawn_rows = int(candidates["Advising Priority Class"].eq(CLASS_WITHDRAWN).sum()) if not candidates.empty else 0
    selectable = set()
    if not candidates.empty:
        selectable_mask = candidates["Candidate Status"].isin([STATUS_ALLOWED, STATUS_ADVISOR])
        incorrect_allowed = int((selectable_mask & candidates["Prerequisite Eligibility"].eq(NOT_ELIGIBLE)).sum())
        incorrect_manual = int((selectable_mask & ((candidates["Eligibility Evidence Status"].eq(EVIDENCE_INSUFFICIENT)) | (candidates["Candidate Reason"].eq("Prerequisite evidence requires manual review.")))).sum())
        conditional_definite = int((candidates["Advising Priority Class"].eq(CLASS_CONDITIONAL) & selectable_mask).sum())
        unknown_candidates = int(candidates["Student Code"].isin(set(state.loc[unknown, "Student Code"])).sum())
        conflict_students = set(state.loc[state["Mixing + Repeat Conflict"].eq(REVIEW_YES), "Student Code"])
        conflict_gap = int((candidates["Student Code"].isin(conflict_students) & candidates["Mixing Required"].eq(REVIEW_YES) & candidates["Advisor Approval Required"].ne(REVIEW_YES) & candidates["Candidate Status"].isin([STATUS_ALLOWED, STATUS_ADVISOR, STATUS_MANUAL])).sum())
        student_gap = int((state["Mixing + Repeat Conflict"].eq(REVIEW_YES) & state["Advisor Review Required"].ne(REVIEW_YES)).sum())
        missing_credit = candidates["Credit Hours"].map(_credit_value).isna()
        missing_selectable = int((missing_credit & selectable_mask).sum())
        missing_other = int((missing_credit & ~selectable_mask).sum())
        duplicate_key = ["Student Code", "Course Code", "Conditional Pathway", "Requirement Scope"]
        duplicates = int(candidates.duplicated(duplicate_key).sum())
    else:
        incorrect_allowed = incorrect_manual = conditional_definite = unknown_candidates = conflict_gap = student_gap = missing_selectable = missing_other = duplicates = 0
    lcgpa = {_text(row["Student Code"]): _number(row.get("Latest LCGPA")) for _, row in profiles.iterrows()} if "Latest LCGPA" in profiles.columns else {}
    outside = int(state.apply(lambda row: _outside_limits(row, parsed, lcgpa.get(_text(row["Student Code"]))), axis=1).sum()) if total else 0
    unimplemented = int(parsed.audit["Implemented"].ne(REVIEW_YES).sum())
    missing_prerequisite = 0 if candidates.empty else int(candidates["Candidate Reason"].eq("Prerequisite result is missing for this remaining course.").sum())
    del prerequisites
    checks = [
        _check("Total students", total, fail=total != profile_students),
        _check("Unique students", unique, fail=unique != total or unique != profile_students),
        _check("Normal students", int(normal.sum()), fail=int(normal.sum()) + int(probation.sum()) != total),
        _check("Probation students", int(probation.sum()), fail=int((profiles["Probation Status"].eq("Probation")).sum()) != int(probation.sum()) if "Probation Status" in profiles.columns else False),
        _check("Students with max 4-course probation limit", probation_limit, fail=int((probation & ~unknown).sum()) != probation_limit or int((unknown & state["Maximum Courses"].notna()).sum()) != 0),
        _check("Students with normal 4-6 range", normal_range),
        _check("Students eligible for mixing", mixing_yes),
        _check("Students blocked from mixing", mixing_no),
        _check("Students requiring mixing review", mixing_review, review=mixing_review > 0),
        _check("Failed required candidate rows", failed_rows),
        _check("Withdrawn required candidate rows", withdrawn_rows),
        _check("Not-eligible courses incorrectly allowed", incorrect_allowed, fail=incorrect_allowed > 0),
        _check("Manual-review prerequisite rows incorrectly allowed", incorrect_manual, fail=incorrect_manual > 0),
        _check("Conditional-pathway courses incorrectly definite", conditional_definite, fail=conditional_definite > 0),
        _check("Unknown-path student assigned course candidates", unknown_candidates, fail=unknown_candidates > 0),
        _check("Repeat + mixing cases without required advisor approval", conflict_gap + student_gap, fail=conflict_gap + student_gap > 0),
        _check("Courses missing credit hours", missing_selectable + missing_other, fail=missing_selectable > 0, review=missing_selectable == 0 and missing_other > 0),
        _check("Course loads outside source limits", outside, fail=outside > 0),
        _check("Spring 2026 leakage", leakage, fail=leakage > 0),
        _check("Duplicate student-course candidate rows", duplicates, fail=duplicates > 0),
        _check("Official advising rules not represented", unimplemented, fail=unimplemented > 0),
        _check("Remaining courses without a prerequisite result", missing_prerequisite, fail=missing_prerequisite > 0),
    ]
    return pd.DataFrame(checks)


def _outside_limits(row: pd.Series, parsed: ParsedAdvisingRules, lcgpa: float | None) -> bool:
    if row["Advising Status"] == ADVISING_PATHWAY or row["Pathway Readiness"] == READINESS_UNKNOWN:
        return any(_present(row[column]) for column in ("Minimum Courses", "Maximum Courses", "Minimum Credits", "Maximum Credits"))
    expected = course_load_envelope(
        parsed,
        probation=row["Probation Status"] == "Probation",
        mixing_allowed=_text(row["Mixing Allowed"]),
        lcgpa=lcgpa,
    )
    for column in ("Minimum Courses", "Maximum Courses", "Minimum Credits", "Maximum Credits"):
        if _number(row[column]) != _number(expected[column]):
            return True
    return False


def _student_directory(profiles: pd.DataFrame, remaining: pd.DataFrame, summary: pd.DataFrame | None, mapping: pd.DataFrame | None) -> list[dict[str, object]]:
    _require_columns(profiles, ["Student Code", "Current Level", "Probation Status"], "Student academic profiles")
    records = []
    summary_index = _index_frame(summary, "Student Code") if summary is not None else {}
    mapping_index = _index_frame(mapping, "Student Code") if mapping is not None else {}
    remaining_index: dict[str, dict[str, object]] = {}
    if not remaining.empty:
        for _, row in remaining.iterrows():
            remaining_index.setdefault(_text(row["Student Code"]), row.to_dict())
    for _, profile in profiles.sort_values("Student Code", kind="mergesort").iterrows():
        student = _text(profile["Student Code"])
        source = summary_index.get(student) or remaining_index.get(student) or {}
        mapped = mapping_index.get(student, {})
        pathway = _text(source.get("Assigned Pathway")) or _text(mapped.get("Assigned Pathway"))
        if mapped and _text(mapped.get("Assigned Pathway")) and pathway and _text(mapped.get("Assigned Pathway")) != pathway:
            raise ValueError(f"{student} has a different assigned pathway in the study-plan mapping and the remaining-course summary.")
        readiness = _text(source.get("Pathway Readiness")) or _text(source.get("Remaining-Course Readiness"))
        records.append({
            "student": student,
            "level": _text(profile["Current Level"]),
            "pathway": pathway,
            "readiness": readiness,
            "possible": _text(source.get("Possible Pathways")),
            "probation": _text(profile["Probation Status"]) == "Probation",
            "probation_status": _text(profile["Probation Status"]),
            "risk": _text(profile.get("Academic Risk Category")) or _text(profile.get("Academic Risk")),
            "lcgpa": _number(profile.get("Latest LCGPA")),
            "cgpa": _number(profile.get("Latest CGPA")),
        })
    return records


def _guard_previous_outputs(profiles: pd.DataFrame, mapping: pd.DataFrame | None, resolution: pd.DataFrame | None, summary: pd.DataFrame | None) -> None:
    if resolution is None or resolution.empty or summary is None or summary.empty:
        return
    if "Remaining-Course Readiness" not in resolution.columns:
        return
    left = resolution.loc[:, ["Student Code", "Remaining-Course Readiness"]].copy()
    right = summary.loc[:, ["Student Code", "Pathway Readiness"]]
    merged = left.merge(right, on="Student Code", how="left")
    mismatch = merged.loc[merged["Remaining-Course Readiness"].map(_text).ne(merged["Pathway Readiness"].map(_text))]
    if not mismatch.empty:
        student = _text(mismatch.iloc[0]["Student Code"])
        raise ValueError(f"{student} has a resolution readiness that disagrees with the remaining-course output.")
    del mapping


def _historical_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if history is None or history.empty:
        return pd.DataFrame(), 0
    frame = history.copy()
    semester = frame["Semester"].map(_text) if "Semester" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    year = frame["Academic Year"].map(_text) if "Academic Year" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    holdout = semester.str.contains(HOLDOUT_TOKEN, na=False) | year.eq(HOLDOUT_TOKEN)
    return frame.loc[~holdout].copy(), int(holdout.sum())


def _credit_catalog(plan: pd.DataFrame) -> dict[str, float | None]:
    _require_columns(plan, ["Course Code", "Credit Hours"], "Study Plan Courses")
    catalog: dict[str, set[float]] = {}
    for _, row in plan.iterrows():
        code = _code(row["Course Code"])
        credit = _credit_value(row["Credit Hours"])
        if code and credit is not None:
            catalog.setdefault(code, set()).add(credit)
    return {code: next(iter(values)) if len(values) == 1 else None for code, values in catalog.items()}


def _title_catalog(plan: pd.DataFrame) -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = {}
    if "Course Title" not in plan.columns:
        return catalog
    for _, row in plan.iterrows():
        catalog.setdefault(_text(row["Course Title"]).casefold(), set()).add(_code(row["Course Code"]))
    return catalog


def _prerequisite_index(prerequisites: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    index: dict[tuple[str, str, str], dict[str, object]] = {}
    if prerequisites is None or prerequisites.empty:
        return index
    for _, row in prerequisites.iterrows():
        code = _code(row.get("Course Code"))
        if code == "":
            continue
        key = (_text(row.get("Student Code")), code, _text(row.get("Conditional Pathway")))
        if key in index:
            index[key] = {}
        else:
            index[key] = row.to_dict()
    return index


def _repeat_index(history: pd.DataFrame, profiles: pd.DataFrame, remaining: pd.DataFrame) -> dict[str, dict[str, object]]:
    failed_remaining: dict[str, set[str]] = {}
    remaining_codes: dict[str, set[str]] = {}
    if not remaining.empty:
        for _, row in remaining.iterrows():
            student = _text(row["Student Code"])
            code = _code(row.get("Course Code"))
            if code == "":
                continue
            remaining_codes.setdefault(student, set()).add(code)
            if _text(row.get("Remaining Reason")) == REASON_FAILED and _text(row.get("Remaining Status")) == STATUS_REMAINING:
                failed_remaining.setdefault(student, set()).add(code)
    failure: dict[str, set[str]] = {}
    improvement: dict[str, set[str]] = {}
    withdrawal: dict[str, set[str]] = {}
    unresolved_codes: dict[str, set[str]] = {}
    if not history.empty:
        derived = derive_repeat_classification(history)
        for _, row in derived.iterrows():
            student = _text(row.get("Student Code"))
            code = _code(row.get("Course Code"))
            repeat_type = _text(row.get("Derived Repeat Type"))
            if repeat_type == DERIVED_FAILURE:
                failure.setdefault(student, set()).add(code)
            elif repeat_type == DERIVED_IMPROVEMENT:
                improvement.setdefault(student, set()).add(code)
            elif repeat_type == DERIVED_WITHDRAWAL:
                withdrawal.setdefault(student, set()).add(code)
            elif repeat_type == DERIVED_UNRESOLVED:
                unresolved_codes.setdefault(student, set()).add(code)
    listed: dict[str, set[str]] = {}
    if "Repeated Courses" in profiles.columns:
        for _, row in profiles.iterrows():
            listed[_text(row["Student Code"])] = _split_codes(row.get("Repeated Courses"))
    students = set(profiles["Student Code"].map(_text))
    index: dict[str, dict[str, object]] = {}
    for student in students:
        types: list[str] = []
        active = bool(failed_remaining.get(student))
        if active or failure.get(student):
            types.append(REPEAT_FAILED)
        if withdrawal.get(student):
            types.append(REPEAT_WITHDRAWAL)
        if improvement.get(student):
            types.append(REPEAT_IMPROVE)
        unresolved = (listed.get(student, set()) | unresolved_codes.get(student, set())) - failure.get(student, set()) - improvement.get(student, set()) - withdrawal.get(student, set())
        outstanding = unresolved_codes.get(student, set()) & remaining_codes.get(student, set())
        if outstanding or (unresolved and not types):
            types.append(REPEAT_UNRESOLVED)
        if outstanding:
            active = True
        index[student] = {
            "present": REVIEW_YES if types else REVIEW_NO,
            "type": "; ".join(dict.fromkeys(types)),
            "active": active,
            "improvement": improvement.get(student, set()),
        }
    return index


def _withdrawal_index(history: pd.DataFrame, limits: dict[str, int]) -> dict[str, dict[str, object]]:
    counts: dict[str, dict[str, int]] = {}
    unresolved: set[str] = set()
    if not history.empty and "Is Withdrawn" in history.columns:
        for _, row in history.iterrows():
            if not _flag(row.get("Is Withdrawn")):
                continue
            student = _text(row.get("Student Code"))
            level = _text(row.get("Level Derived"))
            if level not in limits:
                unresolved.add(student)
                continue
            counts.setdefault(student, {})
            counts[student][level] = counts[student].get(level, 0) + 1
    index: dict[str, dict[str, object]] = {}
    students = set(counts) | unresolved
    for student in students:
        recorded = counts.get(student, {})
        exceeded = any(recorded.get(level, 0) > allowance for level, allowance in limits.items()) or student in unresolved
        index[student] = {
            "allowance": "; ".join(f"{level}: {allowance}" for level, allowance in limits.items()),
            "recorded": "; ".join(f"{level}: {recorded.get(level, 0)}" for level in limits) + ("; Unassigned level" if student in unresolved else ""),
            "exceeded": exceeded,
        }
    return index


def _specialization_index(history: pd.DataFrame, students: list[dict[str, object]], titles: dict[str, set[str]], parsed: ParsedAdvisingRules) -> dict[str, str]:
    if not parsed.grade_conditions or parsed.conditions_required is None:
        return {}
    grades: dict[tuple[str, str], str] = {}
    if not history.empty:
        ordered = history.copy()
        ordered["_year"] = ordered["Academic Year"].map(_year_value) if "Academic Year" in ordered.columns else 0
        ordered["_term"] = ordered["Term"].map(_term_value) if "Term" in ordered.columns else 0
        ordered["_attempt"] = ordered["Attempt Number"].map(_attempt_value) if "Attempt Number" in ordered.columns else 1
        ordered = ordered.sort_values(["_year", "_term", "_attempt"], kind="mergesort")
        for _, row in ordered.iterrows():
            if not _flag(row.get("Is Passed")):
                continue
            grade = normalize_grade(row.get("Grade"))
            if grade is None:
                continue
            grades[(_text(row.get("Student Code")), _code(row.get("Course Code")))] = grade
    results: dict[str, str] = {}
    for profile in students:
        student = _text(profile["student"])
        possible = _text(profile.get("possible"))
        assigned = _text(profile.get("pathway"))
        if PATH_DSAI.casefold() not in possible.casefold() and assigned != PATH_DSAI:
            results[student] = "Not applicable. The possible pathways do not include Data Science and Artificial Intelligence."
            continue
        met = 0
        unresolved = 0
        for condition in parsed.grade_conditions:
            codes = titles.get(condition.title.casefold(), set())
            if not codes:
                unresolved += 1
                continue
            grade = next((grades[(student, code)] for code in codes if (student, code) in grades), "")
            if grade == "":
                continue
            if grade not in GRADE_RANK or condition.minimum not in GRADE_RANK:
                unresolved += 1
                continue
            if GRADE_RANK[grade] >= GRADE_RANK[condition.minimum]:
                met += 1
        cgpa = _number(profile.get("cgpa"))
        if parsed.cgpa_minimum is None or cgpa is None:
            unresolved += 1
        elif cgpa >= parsed.cgpa_minimum:
            met += 1
        required = parsed.conditions_required
        total = len(parsed.grade_conditions) + (1 if parsed.cgpa_minimum is not None else 0)
        if met >= required:
            outcome = "Met"
        elif met + unresolved < required:
            outcome = "Not met"
        else:
            outcome = "Unresolved"
        results[student] = (
            f"{outcome}. {met} of {total} stated conditions are met; the rule requires any {required}. "
            "This evidence does not assign a pathway."
        )
    return results


def _empty_repeat() -> dict[str, object]:
    return {"present": REVIEW_NO, "type": "", "active": False, "improvement": set()}


def _empty_withdrawal(limits: dict[str, int]) -> dict[str, object]:
    return {
        "allowance": "; ".join(f"{level}: {allowance}" for level, allowance in limits.items()),
        "recorded": "; ".join(f"{level}: 0" for level in limits),
        "exceeded": False,
    }


def _group_rows(frame: pd.DataFrame | None, key: str) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    if frame is None or frame.empty or key not in frame.columns:
        return groups
    for row in frame.to_dict(orient="records"):
        groups.setdefault(_text(row.get(key)), []).append(row)
    return groups


def _slot_counts(slots: list[dict[str, object]]) -> tuple[int, int, int, int]:
    ambiguous = future = conditional_future = specialization = 0
    for slot in slots:
        pending = int(_number(slot.get("Pending Slots")) or 0)
        conditional = _text(slot.get("Conditional")).casefold() in {"yes", "true", "y"}
        scope = _text(slot.get("Requirement Scope"))
        if scope == SCOPE_FUTURE and conditional:
            conditional_future += pending
        elif scope == SCOPE_FUTURE:
            future += pending
        elif conditional or scope == SCOPE_SPECIFIC:
            specialization += pending
        elif scope in {SCOPE_CURRENT, SCOPE_COMMON}:
            ambiguous += pending
    return ambiguous, future, conditional_future, specialization


def _level_status(student_level: str, plan_level: str, scope: str, remaining_status: str) -> str:
    del remaining_status
    if scope == SCOPE_FUTURE:
        return LEVEL_FUTURE
    if plan_level == PLACEMENT_MISSING:
        return LEVEL_UNRESOLVED
    if student_level not in LEVEL_RANK:
        if scope == SCOPE_COMMON and plan_level == LEVEL_DIPLOMA:
            return LEVEL_CURRENT
        return LEVEL_UNRESOLVED
    parts = [part.strip() for part in plan_level.split("|")] if plan_level else []
    if not parts or any(part not in LEVEL_RANK for part in parts):
        return LEVEL_UNRESOLVED
    ranks = [LEVEL_RANK[part] for part in parts]
    if any(rank > LEVEL_RANK[student_level] for rank in ranks):
        return LEVEL_FUTURE
    if all(rank < LEVEL_RANK[student_level] for rank in ranks):
        return LEVEL_PREVIOUS
    if all(rank == LEVEL_RANK[student_level] for rank in ranks):
        return LEVEL_CURRENT
    return LEVEL_UNRESOLVED


def _mixing(allowed: str, reason: str, *, conflict: bool, advisor: bool) -> dict[str, object]:
    return {"allowed": allowed, "reason": reason, "conflict": conflict, "advisor": advisor}


def _manual_prerequisite(row: dict[str, object]) -> bool:
    return row["Eligibility Evidence Status"] == EVIDENCE_INSUFFICIENT or row["Candidate Reason"] == "Prerequisite evidence requires manual review."


def _parse_course_load(row: pd.Series | None) -> tuple[int, int, int, int] | None:
    if row is None:
        return None
    courses = re.search(r"(\d+)\s+to\s+(\d+)\s+courses", _text(row["Rule"]), flags=re.IGNORECASE)
    credits = re.search(r"(\d+)\s+to\s+(\d+)\s+credits", _text(row["Rule"]), flags=re.IGNORECASE)
    if not courses or not credits:
        return None
    return int(courses.group(1)), int(courses.group(2)), int(credits.group(1)), int(credits.group(2))


def _parse_probation(row: pd.Series | None) -> tuple[int, int, bool] | None:
    if row is None:
        return None
    match = re.search(r"maximum\s+(\d+)\s+courses\s*/\s*(\d+)\s+credits", _text(row["Rule"]), flags=re.IGNORECASE)
    if not match:
        return None
    exact = re.search(r"only\s+(\d+)\s+credit", _text(row["Source/Notes"]), flags=re.IGNORECASE)
    credits = int(match.group(2))
    exact_match = bool(exact and int(exact.group(1)) == credits)
    return int(match.group(1)), credits, exact_match


def _parse_mixing_threshold(row: pd.Series | None) -> int | None:
    if row is None:
        return None
    match = re.search(r"(\d+)\s+or fewer", _text(row["Rule"]), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_mixing_load(row: pd.Series | None) -> tuple[int, float, int] | None:
    if row is None:
        return None
    normal = re.search(r"normally\s+(\d+)\s+courses", _text(row["Rule"]), flags=re.IGNORECASE)
    extra = re.search(r"LCGPA is\s+(\d+(?:\.\d+)?)\s+or above,\s+it can be\s+(\d+)\s+courses", _text(row["Rule"]), flags=re.IGNORECASE)
    if not normal or not extra:
        return None
    return int(normal.group(1)), float(extra.group(1)), int(extra.group(2))


def _parse_withdrawal_limits(row: pd.Series | None) -> dict[str, int]:
    if row is None:
        return {}
    text = _text(row["Rule"])
    diploma = re.search(r"(one|two|three|\d+)\s+withdrawals?\s+per\s+level\s+for\s+Diploma\s+and\s+Advanced\s+Diploma", text, flags=re.IGNORECASE)
    bachelor = re.search(r"(one|two|three|\d+)\s+withdrawals?\s+in\s+Bachelor", text, flags=re.IGNORECASE)
    if not diploma or not bachelor:
        return {}
    return {
        LEVEL_DIPLOMA: _word_number(diploma.group(1)),
        LEVEL_ADVANCED: _word_number(diploma.group(1)),
        LEVEL_BACHELOR: _word_number(bachelor.group(1)),
    }


def _parse_dsai(row: pd.Series | None) -> tuple[tuple[GradeCondition, ...], int, float] | None:
    if row is None:
        return None
    text = _text(row["Rule"])
    if "any two" not in text.casefold():
        return None
    found = re.findall(r"at least\s+([A-F][+-]?)\s+in\s+([^,]+)", text, flags=re.IGNORECASE)
    cgpa = re.search(r"CGPA\s+(\d+(?:\.\d+)?)\s+or above", text, flags=re.IGNORECASE)
    if len(found) < 2 or not cgpa:
        return None
    conditions = tuple(GradeCondition(minimum=grade.upper(), title=title.strip()) for grade, title in found)
    return conditions, 2, float(cgpa.group(1))


def _parse_foundation_prefix(row: pd.Series | None) -> str:
    if row is None:
        return ""
    match = re.search(r"starting with\s+([A-Z]{2,})", _text(row["Rule"]), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _word_number(value: str) -> int:
    key = value.casefold()
    return _NUMBER_WORDS[key] if key in _NUMBER_WORDS else int(value)


def _code(value: object) -> str:
    normalized = normalize_course_code(value)
    try:
        if pd.isna(normalized):
            return ""
    except TypeError:
        if normalized is None:
            return ""
    return str(normalized)


def _split_codes(value: object) -> set[str]:
    return {code for part in re.split(r"[;,]", _text(value)) if (code := _code(part))}


def _index_frame(frame: pd.DataFrame | None, key: str) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or key not in frame.columns:
        return {}
    return {_text(row[key]): row.to_dict() for _, row in frame.iterrows()}


def _frame(rows: list[dict[str, object]], columns: list[str], sort_by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    ordered = [column for column in sort_by if column in frame.columns]
    return frame.sort_values(ordered, kind="mergesort").reset_index(drop=True)


def _check(check: str, result: object, *, fail: bool = False, review: bool = False) -> dict[str, object]:
    if fail:
        status = "FAIL"
    elif review:
        status = "REVIEW"
    else:
        status = "PASS"
    return {"Check": check, "Result": result, "Status": status}


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: " + ", ".join(missing))


def _flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1"}
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def _text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        if value is None:
            return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def _present(value: object) -> bool:
    return _text(value) != ""


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _credit_value(value: object) -> float | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return number


def _year_value(value: object) -> int:
    digits = re.findall(r"\d{4}", _text(value))
    return int(digits[0]) if digits else 0


def _term_value(value: object) -> int:
    return 0 if "spring" in _text(value).casefold() else 1


def _attempt_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _format(worksheet, widths: dict[str, int], *, wrap_columns: set[str] | None = None) -> None:
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.border = Border(bottom=Side(style="thin", color="666666"))
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    worksheet.freeze_panes = "B2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.row_dimensions[1].height = 22
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width
    wrap_columns = wrap_columns or set()
    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
        for cell in row:
            if get_column_letter(cell.column) in wrap_columns:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
