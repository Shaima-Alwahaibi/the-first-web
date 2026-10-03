"""Identify remaining study-plan requirements from a known pathway scope.

Pathway decisions come from the mapping and resolution workbooks. This module
does not choose a pathway, check prerequisites, or read Spring 2026.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from preparation_checks import normalize_course_code, write_atomic
from study_plan_matching import (
    HOLDOUT_TOKEN,
    LEVEL_ADVANCED,
    LEVEL_BACHELOR,
    LEVEL_DIPLOMA,
    PATH_COMMON,
    PATH_CYBER,
    PATH_DSAI,
    PATH_NCS,
    PATH_REVIEW,
    PATH_SE,
    READINESS_FULL,
    READINESS_PARTIAL,
    READINESS_UNKNOWN,
    SOURCE_NCS_YEAR2,
    SOURCE_SE_YEAR2,
)

STATUS_PATHWAY_REQUIRED = "Pathway Resolution Required"
STATUS_REMAINING = "Remaining"
STATUS_COMPLETED = "Completed"
STATUS_CONDITIONAL = "Conditional"

SCOPE_CURRENT = "Current/Previous Level"
SCOPE_FUTURE = "Future Level"
SCOPE_COMMON = "Common Across Possible Paths"
SCOPE_SPECIFIC = "Specialization Dependent"

REASON_FAILED = "Failed Required Course"
REASON_WITHDRAWN = "Withdrawn Required Course"
REASON_NOT_TAKEN = "Not Yet Taken"
REASON_SATISFIED = "Requirement Satisfied"
REASON_CONDITIONAL = "Conditional on the selected specialization"
REASON_UNKNOWN = STATUS_PATHWAY_REQUIRED
REASON_ELECTIVE = "Named Major Elective Candidate"
GENERAL_REQUIREMENT_POOL = "General Requirement"
PLACEMENT_MISSING = "Placement Evidence Missing"

HISTORICAL_PASSED = "Passed"
HISTORICAL_FAILED = "Failed"
HISTORICAL_WITHDRAWN = "Withdrawn"
HISTORICAL_NOT_TAKEN = "Not Taken"

LEVEL_RANK = {LEVEL_DIPLOMA: 1, LEVEL_ADVANCED: 2, LEVEL_BACHELOR: 3}
PATHWAY_BLOCKS = {
    PATH_COMMON: ((PATH_COMMON, LEVEL_DIPLOMA),),
    PATH_NCS: ((PATH_COMMON, LEVEL_DIPLOMA), (SOURCE_NCS_YEAR2, LEVEL_DIPLOMA)),
    PATH_SE: (
        (PATH_COMMON, LEVEL_DIPLOMA),
        (SOURCE_SE_YEAR2, LEVEL_DIPLOMA),
        (PATH_SE, LEVEL_ADVANCED),
        (PATH_SE, LEVEL_BACHELOR),
    ),
    PATH_DSAI: (
        (PATH_COMMON, LEVEL_DIPLOMA),
        (SOURCE_SE_YEAR2, LEVEL_DIPLOMA),
        (PATH_DSAI, LEVEL_ADVANCED),
        (PATH_DSAI, LEVEL_BACHELOR),
    ),
    PATH_CYBER: (
        (PATH_COMMON, LEVEL_DIPLOMA),
        (SOURCE_NCS_YEAR2, LEVEL_DIPLOMA),
        (PATH_CYBER, LEVEL_ADVANCED),
        (PATH_CYBER, LEVEL_BACHELOR),
    ),
}
FUTURE_CONTINUATION = {
    PATH_NCS: ((PATH_CYBER, LEVEL_ADVANCED), (PATH_CYBER, LEVEL_BACHELOR)),
}

REMAINING_COLUMNS = [
    "Student Code", "Pathway Readiness", "Assigned Pathway", "Possible Pathways", "Current Level",
    "Course Code", "Course Title", "Study Plan Level", "Study Plan Year", "Course Type",
    "Requirement Type", "Is Core", "Is Elective", "Elective Pool", "Requirement Scope",
    "Historical Status", "Remaining Status", "Remaining Reason", "Conditional Pathway",
    "Credit Hours",
]
SUMMARY_COLUMNS = [
    "Student Code", "Pathway Readiness", "Assigned Pathway", "Possible Pathways", "Current Level",
    "Completed Core Count", "Remaining Core Count", "Failed Required Count", "Withdrawn Required Count",
    "Not Yet Taken Core Count", "Completed Elective Count", "Pending Elective Slots",
    "Common Remaining Count", "Conditional Remaining Count", "Remaining Course Status",
    "Pending Named Elective Slots",
]
ELECTIVE_COLUMNS = [
    "Student Code", "Pathway", "Elective Pool", "Required Slots", "Completed Slots", "Pending Slots",
    "Completed Elective Codes", "Conditional", "Requirement Scope", "Notes",
]
AUDIT_COLUMNS = [
    "Student Code", "Course Code", "Official Requirement", "Historical Attempts",
    "Final Historical Status", "Requirement Decision", "Decision Reason",
]


def build_pathway_requirements(plan: pd.DataFrame, pools: pd.DataFrame) -> dict[str, object]:
    """Build core courses and elective slots from the official study-plan rows."""
    work = plan.copy()
    work["Course Code"] = work["Course Code"].map(normalize_course_code)
    cores: dict[tuple[str, str], list[dict[str, object]]] = {}
    slots: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in work.to_dict(orient="records"):
        code = record.get("Course Code")
        source = _text(record.get("Specialization Path"))
        level = _text(record.get("Level"))
        if not source or not level or _missing(code):
            continue
        key = (source, level)
        if _yes(record.get("Is Core")):
            cores.setdefault(key, []).append(_course_record(record, core=True))
        elif _yes(record.get("Is Elective")):
            slots.setdefault(key, []).append(_course_record(record, core=False))
    pool_courses: dict[str, set[str]] = {}
    pool_details: dict[str, dict[str, dict[str, object]]] = {}
    pool_frame = pools.copy()
    pool_frame["Course Code"] = pool_frame["Course Code"].map(normalize_course_code)
    for record in pool_frame.to_dict(orient="records"):
        code = record.get("Course Code")
        pool = _text(record.get("Elective Pool"))
        if not pool or _missing(code) or str(code).upper().startswith("FP"):
            continue
        pool_courses.setdefault(pool, set()).add(str(code))
        pool_details.setdefault(pool, {})[str(code)] = {
            "title": _text(record.get("Course Title")),
            "credits": record.get("Credit Hours"),
        }
    placement: dict[str, dict[str, str]] = {}
    observed: dict[str, set[tuple[str, str]]] = {}
    for record in work.to_dict(orient="records"):
        code = record.get("Course Code")
        if _missing(code):
            continue
        label = str(code)
        if label.casefold().startswith("major elective") or label.casefold().startswith("general requirement"):
            continue
        level = _text(record.get("Level"))
        year = _text(record.get("Year"))
        if level:
            observed.setdefault(label, set()).add((level, year))
    for code, pairs in observed.items():
        levels = {level for level, _year in pairs}
        years = {year for _level, year in pairs}
        if len(levels) == 1 and len(years) == 1:
            level, year = next(iter(pairs))
            placement[code] = {"level": level, "year": year}
    return {"cores": cores, "slots": slots, "pools": pool_courses, "pool_details": pool_details, "placement": placement}


def build_student_course_history(transcript: pd.DataFrame) -> dict[str, object]:
    """Collapse historical attempts. A later pass replaces an earlier failure or withdrawal."""
    required = {"Student Code", "Semester", "Academic Year", "Course Code", "Is Passed", "Is Failed", "Is Withdrawn"}
    missing = required - set(transcript.columns)
    if missing:
        raise RuntimeError("Historical transcript is missing columns: " + ", ".join(sorted(missing)))
    holdout = transcript["Semester"].astype(str).str.contains(HOLDOUT_TOKEN) | transcript["Academic Year"].astype(str).str.contains(HOLDOUT_TOKEN)
    leakage = int(holdout.sum())
    history = transcript.loc[~holdout].copy()
    by_student: dict[str, dict[str, dict[str, object]]] = {}
    fp_rows = 0
    for record in history.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code):
            continue
        code = str(code)
        if code.startswith("FP"):
            fp_rows += 1
            continue
        student = str(record["Student Code"])
        attempt = {
            "order": (_year_value(record.get("Academic Year")), _term_value(record.get("Semester")), _attempt_value(record.get("Attempt Number"))),
            "label": f"{_text(record.get('Academic Year'))} {_term_name(record.get('Semester'))}: {_attempt_label(record)}",
            "passed": _flag(record.get("Is Passed")),
            "failed": _flag(record.get("Is Failed")),
            "withdrawn": _flag(record.get("Is Withdrawn")),
            "improvement": _flag(record.get("Is Repeated For Improvement")) or "N" == _text(record.get("Remarks")).upper(),
        }
        by_student.setdefault(student, {}).setdefault(code, {"attempts": []})["attempts"].append(attempt)
    final: dict[str, dict[str, dict[str, object]]] = {}
    for student, courses in by_student.items():
        final[student] = {}
        for code, payload in courses.items():
            attempts = sorted(payload["attempts"], key=lambda item: item["order"])
            final[student][code] = {
                "status": _final_status(attempts),
                "attempts": "; ".join(item["label"] for item in attempts),
                "improvement": any(item["improvement"] and item["passed"] for item in attempts),
            }
    return {"students": final, "spring_2026_leakage": leakage, "fp_rows_excluded": fp_rows}


def build_readiness(mapping: pd.DataFrame, resolution: pd.DataFrame) -> pd.DataFrame:
    """Read pathway readiness from the completed mapping outputs."""
    resolved = resolution.set_index(resolution["Student Code"].astype(str))
    rows = []
    for record in mapping.to_dict(orient="records"):
        student = str(record["Student Code"])
        assigned = _text(record.get("Assigned Pathway"))
        level = _text(record.get("Current Level"))
        if _text(record.get("Manual Review Required")) == "No":
            readiness = READINESS_FULL
            possible = assigned
            base = pd.NA
        else:
            if student not in resolved.index:
                raise RuntimeError(f"{student} is manual review and has no resolution row.")
            source = resolved.loc[student]
            readiness = _text(source.get("Remaining-Course Readiness"))
            possible = _text(source.get("Possible Pathways"))
            base = source.get("Year 2 Base Path")
            if readiness not in {READINESS_PARTIAL, READINESS_UNKNOWN}:
                raise RuntimeError(f"{student} has an unexpected readiness value: {readiness}")
        rows.append({
            "Student Code": student,
            "Pathway Readiness": readiness,
            "Assigned Pathway": assigned,
            "Possible Pathways": possible,
            "Current Level": level,
            "Year 2 Base Path": base,
        })
    frame = pd.DataFrame(rows).sort_values("Student Code", kind="mergesort").reset_index(drop=True)
    if not frame["Student Code"].is_unique:
        raise RuntimeError("Pathway readiness has duplicate Student Codes.")
    return frame


def build_remaining_course_output(
    readiness: pd.DataFrame,
    requirements: dict[str, object],
    history: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Compare each student's Fall 2025 history with the pathway scope already decided."""
    remaining_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    elective_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for record in readiness.to_dict(orient="records"):
        student = str(record["Student Code"])
        courses = history["students"].get(student, {})
        package = _requirements_for_student(record, requirements)
        if package["unknown"]:
            summary_rows.append(_unknown_summary(record))
            remaining_rows.append(_unknown_remaining_row(record))
            continue
        core_results = [_decide_core(student, item, courses) for item in package["cores"]]
        _add_common_year2(student, record, package, courses, requirements, core_results)
        elective_results = _evaluate_electives(student, package["slots"], courses, package["blocked_codes"], requirements["pools"])
        named_rows, named_pending = _named_major_electives(
            student, record, elective_results["rows"], courses, requirements, package["blocked_codes"],
        )
        audit_rows.extend(item["audit"] for item in core_results)
        remaining_rows.extend(item["row"] for item in core_results if item["row"] is not None)
        remaining_rows.extend(named_rows)
        elective_rows.extend(elective_results["rows"])
        summary_rows.append(_summary_row(record, core_results, elective_results, named_pending))
    remaining = _frame(remaining_rows, REMAINING_COLUMNS, ["Student Code", "Conditional Pathway", "Course Code"])
    summary = _frame(summary_rows, SUMMARY_COLUMNS, ["Student Code"])
    electives = _frame(elective_rows, ELECTIVE_COLUMNS, ["Student Code", "Conditional", "Elective Pool"])
    audit = _frame(audit_rows, AUDIT_COLUMNS, ["Student Code", "Course Code"])
    validation = validate_remaining_courses(summary, remaining, electives, audit, readiness, history, requirements)
    return {
        "remaining": remaining,
        "summary": summary,
        "electives": electives,
        "audit": audit,
        "validation": validation,
    }


def validate_remaining_courses(
    summary: pd.DataFrame,
    remaining: pd.DataFrame,
    electives: pd.DataFrame,
    audit: pd.DataFrame,
    readiness: pd.DataFrame,
    history: dict[str, object],
    requirements: dict[str, object],
) -> pd.DataFrame:
    """Return integrity checks. A broken invariant is FAIL."""
    definite = remaining.loc[remaining["Remaining Status"].eq(STATUS_REMAINING)] if not remaining.empty else remaining
    conditional = remaining.loc[remaining["Remaining Status"].eq(STATUS_CONDITIONAL)] if not remaining.empty else remaining
    partial_ids = set(readiness.loc[readiness["Pathway Readiness"].eq(READINESS_PARTIAL), "Student Code"].astype(str))
    unknown_ids = set(readiness.loc[readiness["Pathway Readiness"].eq(READINESS_UNKNOWN), "Student Code"].astype(str))
    plan_codes = _plan_codes(requirements)
    passed_remaining = 0
    failed_missing = 0
    withdrawn_missing = 0
    if not audit.empty:
        outstanding = set(zip(definite["Student Code"].astype(str), definite["Course Code"].astype(str))) if not definite.empty else set()
        for record in audit.to_dict(orient="records"):
            if record["Requirement Decision"] not in {STATUS_REMAINING, STATUS_COMPLETED}:
                continue
            key = (str(record["Student Code"]), str(record["Course Code"]))
            if record["Final Historical Status"] == HISTORICAL_PASSED and key in outstanding:
                passed_remaining += 1
            if record["Final Historical Status"] == HISTORICAL_FAILED and record["Requirement Decision"] == STATUS_REMAINING and key not in outstanding:
                failed_missing += 1
            if record["Final Historical Status"] == HISTORICAL_WITHDRAWN and record["Requirement Decision"] == STATUS_REMAINING and key not in outstanding:
                withdrawn_missing += 1
    specific_marked_definite = 0
    if not definite.empty:
        specific_marked_definite = int(
            definite["Student Code"].astype(str).isin(partial_ids).sum()
            and definite.loc[definite["Student Code"].astype(str).isin(partial_ids), "Requirement Scope"].eq(SCOPE_SPECIFIC).sum()
        )
    unknown_specific = 0
    if not remaining.empty:
        unknown_rows = remaining.loc[remaining["Student Code"].astype(str).isin(unknown_ids)]
        unknown_specific = int(unknown_rows["Course Code"].map(lambda value: _text(value) != "").astype(int).sum()) if not unknown_rows.empty else 0
    duplicate_remaining = 0
    if not remaining.empty:
        keys = remaining.assign(_conditional=remaining["Conditional Pathway"].fillna(""), _code=remaining["Course Code"].fillna(""))
        duplicate_remaining = int(keys.duplicated(["Student Code", "_code", "_conditional", "Requirement Scope"]).sum())
    invalid_codes = 0
    if not remaining.empty:
        coded = remaining.loc[remaining["Course Code"].map(lambda value: _text(value) != "")]
        invalid_codes = int((~coded["Course Code"].astype(str).isin(plan_codes)).sum())
    negative_pending = int(electives["Pending Slots"].lt(0).sum()) if not electives.empty else 0
    over_completed = int(electives["Completed Slots"].gt(electives["Required Slots"]).sum()) if not electives.empty else 0
    checks = [
        _check("Total students", len(summary), fail=len(summary) != len(readiness)),
        _check("Unique students", summary["Student Code"].nunique() if not summary.empty else 0, fail=not summary["Student Code"].is_unique if not summary.empty else True),
        _check("Full-pathway students", int(readiness["Pathway Readiness"].eq(READINESS_FULL).sum())),
        _check("Partial-pathway students", int(readiness["Pathway Readiness"].eq(READINESS_PARTIAL).sum())),
        _check("Pathway-unknown students", int(readiness["Pathway Readiness"].eq(READINESS_UNKNOWN).sum())),
        _check("Students represented in output", summary["Student Code"].nunique() if not summary.empty else 0, fail=set(summary["Student Code"].astype(str)) != set(readiness["Student Code"].astype(str))),
        _check("Duplicate student summary rows", int(summary["Student Code"].duplicated().sum()) if not summary.empty else 0, fail=bool(summary["Student Code"].duplicated().any()) if not summary.empty else False),
        _check("Spring 2026 leakage", int(history["spring_2026_leakage"]), fail=int(history["spring_2026_leakage"]) != 0),
        _check("FP courses excluded from requirements", int(history["fp_rows_excluded"])),
        _check("Passed required courses incorrectly remaining", passed_remaining, fail=passed_remaining != 0),
        _check("Failed required courses missing from remaining", failed_missing, fail=failed_missing != 0),
        _check("Withdrawn required courses missing from remaining", withdrawn_missing, fail=withdrawn_missing != 0),
        _check("Duplicate remaining requirement rows", duplicate_remaining, fail=duplicate_remaining != 0),
        _check("Invalid course codes", invalid_codes, fail=invalid_codes != 0),
        _check("Unknown study-plan course codes", invalid_codes, fail=invalid_codes != 0),
        _check("Negative elective pending slots", negative_pending, fail=negative_pending != 0),
        _check("Elective completed slots above required slots", over_completed, fail=over_completed != 0),
        _check("Partial-path students with specialization-specific course marked definite", specific_marked_definite, fail=specific_marked_definite != 0),
        _check("Unknown-path student with specialization-specific remaining courses", unknown_specific, fail=unknown_specific != 0),
    ]
    return pd.DataFrame(checks, columns=["Check", "Result", "Status"])


def update_mapping_remaining_fields(
    mapping: pd.DataFrame,
    summary: pd.DataFrame,
    remaining: pd.DataFrame,
    electives: pd.DataFrame,
) -> pd.DataFrame:
    """Fill the deferred mapping fields from the remaining-course result."""
    updated = mapping.copy()
    fields = {
        "Remaining Core Courses": {},
        "Completed Electives": {},
        "Pending Electives": {},
        "Conditional Specialization Requirements": {},
    }
    summary_index = summary.set_index(summary["Student Code"].astype(str))
    for student in updated["Student Code"].astype(str):
        status = _text(summary_index.loc[student, "Remaining Course Status"])
        if status == STATUS_PATHWAY_REQUIRED:
            for column in fields:
                fields[column][student] = STATUS_PATHWAY_REQUIRED
            continue
        student_remaining = remaining.loc[remaining["Student Code"].astype(str).eq(student)] if not remaining.empty else remaining
        definite = student_remaining.loc[student_remaining["Remaining Status"].eq(STATUS_REMAINING)] if not student_remaining.empty else student_remaining
        conditional = student_remaining.loc[student_remaining["Remaining Status"].eq(STATUS_CONDITIONAL)] if not student_remaining.empty else student_remaining
        fields["Remaining Core Courses"][student] = _join(definite["Course Code"]) if not definite.empty else "None"
        student_electives = electives.loc[electives["Student Code"].astype(str).eq(student)] if not electives.empty else electives
        definite_electives = student_electives.loc[student_electives["Conditional"].eq("No")] if not student_electives.empty else student_electives
        fields["Completed Electives"][student] = _completed_elective_text(definite_electives)
        fields["Pending Electives"][student] = _pending_elective_text(definite_electives)
        fields["Conditional Specialization Requirements"][student] = _conditional_text(conditional, student_electives)
    for column, values in fields.items():
        updated[column] = updated["Student Code"].astype(str).map(values)
    return updated


def export_remaining_course_outputs(
    tables: Mapping[str, pd.DataFrame],
    mapping: pd.DataFrame,
    evidence: pd.DataFrame,
    pathway_validation: pd.DataFrame,
    remaining_path: Path,
    mapping_path: Path,
) -> None:
    """Write the remaining-course workbook and the updated mapping workbook."""

    def write_remaining(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["remaining"].to_excel(writer, sheet_name="Remaining Courses", index=False)
            tables["summary"].to_excel(writer, sheet_name="Student Remaining Summary", index=False)
            tables["electives"].to_excel(writer, sheet_name="Elective Requirements", index=False)
            tables["audit"].to_excel(writer, sheet_name="Requirement Audit", index=False)
            tables["validation"].to_excel(writer, sheet_name="Remaining Course Validation", index=False)
            _format(writer.book["Remaining Courses"], {"A": 16, "B": 24, "C": 42, "D": 42, "E": 22, "F": 16, "G": 42, "H": 22, "I": 16, "J": 16, "K": 20, "L": 12, "M": 14, "N": 28, "O": 32, "P": 20, "Q": 28, "R": 36, "S": 42})
            _format(writer.book["Student Remaining Summary"], {"A": 16, "B": 24, "C": 42, "D": 42, "E": 22, "O": 36}, integer_columns={"F", "G", "H", "I", "J", "K", "L", "M", "N"})
            _format(writer.book["Elective Requirements"], {"A": 16, "B": 42, "C": 28, "G": 36, "J": 72}, integer_columns={"D", "E", "F"})
            _format(writer.book["Requirement Audit"], {"A": 16, "B": 16, "C": 42, "D": 55, "E": 24, "F": 28, "G": 42})
            _format(writer.book["Remaining Course Validation"], {"A": 78, "B": 14, "C": 12}, integer_columns={"B"})

    def write_mapping(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            mapping.to_excel(writer, sheet_name="Student Study Plan Mapping", index=False)
            pathway_validation.to_excel(writer, sheet_name="Mapping Validation", index=False)
            evidence.to_excel(writer, sheet_name="Mapping Evidence", index=False)
            _format(writer.book["Student Study Plan Mapping"], {"A": 16, "L": 28, "M": 42, "N": 36, "O": 36, "P": 55}, wrap_columns={"M", "N", "O", "P"})
            _format(writer.book["Mapping Validation"], {"A": 62, "B": 14, "C": 12}, integer_columns={"B"})
            _format(writer.book["Mapping Evidence"], {"A": 16, "B": 16, "D": 42})

    write_atomic(remaining_path, write_remaining)
    write_atomic(mapping_path, write_mapping)


def _requirements_for_student(record: Mapping[str, object], requirements: dict[str, object]) -> dict[str, object]:
    readiness = _text(record.get("Pathway Readiness"))
    if readiness == READINESS_UNKNOWN:
        return {"unknown": True, "cores": [], "slots": [], "blocked_codes": set()}
    level = _text(record.get("Current Level"))
    if readiness == READINESS_FULL:
        pathway = _text(record.get("Assigned Pathway"))
        cores, slots = _scoped_requirements(pathway, level, requirements, conditional="")
        return {"unknown": False, "cores": _stamp(cores, record), "slots": slots, "blocked_codes": {item["code"] for item in cores}}
    se_cores, se_slots = _scoped_requirements(PATH_SE, level, requirements, conditional=PATH_SE)
    dsai_cores, dsai_slots = _scoped_requirements(PATH_DSAI, level, requirements, conditional=PATH_DSAI)
    cores = _stamp(_combine_partial_cores(se_cores, dsai_cores), record)
    slots = _combine_partial_slots(se_slots, dsai_slots)
    blocked = {item["code"] for item in se_cores} | {item["code"] for item in dsai_cores}
    return {"unknown": False, "cores": cores, "slots": slots, "blocked_codes": blocked}


def _scoped_requirements(
    pathway: str,
    student_level: str,
    requirements: dict[str, object],
    *,
    conditional: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if pathway not in PATHWAY_BLOCKS:
        raise RuntimeError(f"No official requirement block is defined for {pathway}.")
    cores: dict[str, dict[str, object]] = {}
    slots: list[dict[str, object]] = []
    blocks = list(PATHWAY_BLOCKS[pathway]) + list(FUTURE_CONTINUATION.get(pathway, ()))
    for source, plan_level in blocks:
        scope = _scope(student_level, plan_level, pathway)
        for course in requirements["cores"].get((source, plan_level), []):
            cores.setdefault(course["code"], {**course, "scope": scope, "conditional": conditional})
        for slot in requirements["slots"].get((source, plan_level), []):
            slots.append({**slot, "scope": scope, "conditional": conditional, "pathway": pathway})
    return list(cores.values()), slots


def _combine_partial_cores(se_cores: list[dict[str, object]], dsai_cores: list[dict[str, object]]) -> list[dict[str, object]]:
    se_index = {item["code"]: item for item in se_cores}
    dsai_index = {item["code"]: item for item in dsai_cores}
    combined = []
    for code in sorted(set(se_index) | set(dsai_index)):
        if code in se_index and code in dsai_index:
            combined.append(_merge_common(se_index[code], dsai_index[code]))
        elif code in se_index:
            combined.append({**se_index[code], "scope": SCOPE_SPECIFIC, "conditional": PATH_SE})
        else:
            combined.append({**dsai_index[code], "scope": SCOPE_SPECIFIC, "conditional": PATH_DSAI})
    return combined


def _combine_partial_slots(se_slots: list[dict[str, object]], dsai_slots: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep a slot only when both pathways require the same pool at the same scope."""
    se_groups = _slot_groups(se_slots)
    dsai_groups = _slot_groups(dsai_slots)
    common = []
    specific = []
    for key, rows in se_groups.items():
        other = dsai_groups.get(key, [])
        shared = min(len(rows), len(other))
        for slot in rows[:shared]:
            common.append({
                **slot,
                "conditional": "",
                "pathway": f"{PATH_SE} | {PATH_DSAI}",
                "scope": SCOPE_COMMON if slot["scope"] != SCOPE_FUTURE else SCOPE_FUTURE,
            })
        for slot in rows[shared:]:
            specific.append({**slot, "conditional": PATH_SE, "scope": SCOPE_SPECIFIC})
    for key, rows in dsai_groups.items():
        shared = min(len(rows), len(se_groups.get(key, [])))
        for slot in rows[shared:]:
            specific.append({**slot, "conditional": PATH_DSAI, "scope": SCOPE_SPECIFIC})
    return common + specific


def _decide_core(student: str, requirement: dict[str, object], courses: dict[str, dict[str, object]]) -> dict[str, object]:
    history = courses.get(requirement["code"])
    status = history["status"] if history else HISTORICAL_NOT_TAKEN
    attempts = history["attempts"] if history else ""
    conditional = requirement.get("conditional") or ""
    if conditional:
        decision = STATUS_COMPLETED if status == HISTORICAL_PASSED else STATUS_CONDITIONAL
        reason = REASON_SATISFIED if status == HISTORICAL_PASSED else REASON_CONDITIONAL
    elif status == HISTORICAL_PASSED:
        decision = STATUS_COMPLETED
        reason = REASON_SATISFIED
    else:
        decision = STATUS_REMAINING
        reason = _remaining_reason(status)
    audit = {
        "Student Code": student,
        "Course Code": requirement["code"],
        "Official Requirement": requirement["source"],
        "Historical Attempts": attempts,
        "Final Historical Status": status,
        "Requirement Decision": decision,
        "Decision Reason": reason if decision != STATUS_CONDITIONAL else f"{reason}: {conditional}",
    }
    row = None
    if decision in {STATUS_REMAINING, STATUS_CONDITIONAL}:
        row = {
            "Student Code": student,
            "Pathway Readiness": requirement.get("readiness", ""),
            "Assigned Pathway": requirement.get("assigned", ""),
            "Possible Pathways": requirement.get("possible", ""),
            "Current Level": requirement.get("level_name", ""),
            "Course Code": requirement["code"],
            "Course Title": requirement["title"],
            "Study Plan Level": requirement["level"],
            "Study Plan Year": requirement["year"],
            "Course Type": requirement["course_type"],
            "Requirement Type": requirement["requirement_type"],
            "Is Core": "Yes",
            "Is Elective": "No",
            "Elective Pool": pd.NA,
            "Requirement Scope": requirement["scope"],
            "Historical Status": status,
            "Remaining Status": decision,
            "Remaining Reason": reason if decision == STATUS_REMAINING else _remaining_reason(status),
            "Conditional Pathway": conditional or pd.NA,
            "Credit Hours": requirement.get("credits", pd.NA),
        }
    return {"audit": audit, "row": row, "decision": decision, "status": status, "scope": requirement["scope"], "conditional": conditional}


def _evaluate_electives(
    student: str,
    slots: list[dict[str, object]],
    courses: dict[str, dict[str, object]],
    blocked_codes: set[str],
    pools: dict[str, set[str]],
) -> dict[str, object]:
    used: set[str] = set()
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for slot in slots:
        grouped.setdefault((_text(slot.get("pool")), slot.get("conditional") or "", slot["scope"]), []).append(slot)
    rows = []
    filled_total = 0
    pending_total = 0
    for (pool, conditional, scope), group in sorted(grouped.items()):
        required = len(group)
        available = sorted(
            code for code, payload in courses.items()
            if payload["status"] == HISTORICAL_PASSED and code in pools.get(pool, set()) and code not in blocked_codes and code not in used
        )
        chosen = available[:required]
        used.update(chosen)
        completed = len(chosen)
        pending = required - completed
        note = ""
        if pool.casefold() == GENERAL_REQUIREMENT_POOL.casefold() and pool not in pools:
            note = "General Requirement Elective — Official Named Pool Required"
            chosen = []
            completed = 0
            pending = required
        elif pool not in pools:
            note = "The official Elective Pools sheet does not list courses for this pool. No historical course was counted toward the slot."
            chosen = []
            completed = 0
            pending = required
        filled_total += 0 if conditional else completed
        pending_total += 0 if conditional else pending
        rows.append({
            "Student Code": student,
            "Pathway": group[0].get("pathway") or "",
            "Elective Pool": pool,
            "Required Slots": required,
            "Completed Slots": completed,
            "Pending Slots": pending,
            "Completed Elective Codes": "; ".join(chosen),
            "Conditional": "Yes" if conditional else "No",
            "Requirement Scope": scope,
            "Notes": note,
        })
    return {"rows": rows, "completed": filled_total, "pending": pending_total}


def _add_common_year2(
    student: str,
    record: Mapping[str, object],
    package: dict[str, object],
    courses: dict[str, dict[str, object]],
    requirements: dict[str, object],
    core_results: list[dict[str, object]],
) -> None:
    """Add the Year 2 cores shared by both branches after Common Year 1 is complete."""

    if _text(record.get("Assigned Pathway")) != PATH_COMMON or not package["cores"]:
        return
    if any((courses.get(item["code"]) or {}).get("status") != HISTORICAL_PASSED for item in package["cores"]):
        return
    existing = {item["audit"]["Course Code"] for item in core_results}
    for course in _shared_year2_cores(requirements):
        if course["code"] in existing:
            continue
        if (courses.get(course["code"]) or {}).get("status") == HISTORICAL_PASSED:
            continue
        stamped = _stamp([dict(course)], record)[0]
        core_results.append(_decide_core(student, stamped, courses))


def _shared_year2_cores(requirements: dict[str, object]) -> list[dict[str, object]]:
    se = {item["code"]: item for item in requirements["cores"].get((SOURCE_SE_YEAR2, LEVEL_DIPLOMA), [])}
    ncs = {item["code"]: item for item in requirements["cores"].get((SOURCE_NCS_YEAR2, LEVEL_DIPLOMA), [])}
    shared = []
    for code in sorted(set(se) & set(ncs)):
        course = dict(se[code])
        course["scope"] = SCOPE_COMMON
        course["conditional"] = ""
        course["source"] = f"{SOURCE_SE_YEAR2} | {SOURCE_NCS_YEAR2}"
        shared.append(course)
    return shared


def _named_major_electives(
    student: str,
    record: Mapping[str, object],
    slot_rows: list[dict[str, object]],
    courses: dict[str, dict[str, object]],
    requirements: dict[str, object],
    blocked_codes: set[str],
) -> tuple[list[dict[str, object]], int]:
    """Expand pending major-elective slots into named pool courses."""

    details = requirements.get("pool_details", {})
    if _text(record.get("Pathway Readiness")) == READINESS_PARTIAL:
        return _partial_major_electives(student, record, slot_rows, courses, requirements, blocked_codes)
    rows: list[dict[str, object]] = []
    pending_named = 0
    seen: set[str] = set()
    for slot in slot_rows:
        pool = _text(slot.get("Elective Pool"))
        pending = int(slot.get("Pending Slots") or 0)
        if pending <= 0 or pool not in details or pool.casefold() == GENERAL_REQUIREMENT_POOL.casefold():
            continue
        if _text(slot.get("Conditional")) == "Yes":
            continue
        pending_named += pending
        completed = {code.strip() for code in _text(slot.get("Completed Elective Codes")).split(";") if code.strip()}
        for code in sorted(details[pool]):
            if code in seen or code in completed or code in blocked_codes or (courses.get(code) or {}).get("status") == HISTORICAL_PASSED:
                continue
            seen.add(code)
            rows.append(_elective_candidate_row(student, record, code, pool, details[pool][code], requirements, courses, conditional=""))
    return rows, pending_named


def _partial_major_electives(
    student: str,
    record: Mapping[str, object],
    slot_rows: list[dict[str, object]],
    courses: dict[str, dict[str, object]],
    requirements: dict[str, object],
    blocked_codes: set[str],
) -> tuple[list[dict[str, object]], int]:
    details = requirements.get("pool_details", {})
    se_pool = "SE Major Elective"
    dsai_pool = "DSAI Major Elective"
    se_pending = _pending_for_pool(slot_rows, se_pool)
    dsai_pending = _pending_for_pool(slot_rows, dsai_pool)
    se_codes = set(details.get(se_pool, {}))
    dsai_codes = set(details.get(dsai_pool, {}))
    overlap = se_codes & dsai_codes
    rows: list[dict[str, object]] = []
    if se_pending > 0 and dsai_pending > 0:
        for code in sorted(overlap):
            if code in blocked_codes or (courses.get(code) or {}).get("status") == HISTORICAL_PASSED:
                continue
            meta = details[se_pool][code]
            rows.append(_elective_candidate_row(
                student, record, code, f"{se_pool} | {dsai_pool}", meta, requirements, courses, conditional="",
            ))
    for pool, pending, conditional in ((se_pool, se_pending, PATH_SE), (dsai_pool, dsai_pending, PATH_DSAI)):
        if pending <= 0:
            continue
        for code in sorted(set(details.get(pool, {})) - overlap):
            if code in blocked_codes or (courses.get(code) or {}).get("status") == HISTORICAL_PASSED:
                continue
            rows.append(_elective_candidate_row(
                student, record, code, pool, details[pool][code], requirements, courses, conditional=conditional,
            ))
    named_pending = min(se_pending, dsai_pending) if se_pending and dsai_pending else 0
    return rows, named_pending


def _pending_for_pool(slot_rows: list[dict[str, object]], pool: str) -> int:
    return sum(int(slot.get("Pending Slots") or 0) for slot in slot_rows if _text(slot.get("Elective Pool")) == pool)


def _elective_candidate_row(
    student: str,
    record: Mapping[str, object],
    code: str,
    pool: str,
    meta: Mapping[str, object],
    requirements: dict[str, object],
    courses: dict[str, dict[str, object]],
    *,
    conditional: str,
) -> dict[str, object]:
    placement = requirements.get("placement", {}).get(code)
    history = courses.get(code)
    status = history["status"] if history else HISTORICAL_NOT_TAKEN
    if placement:
        level = placement["level"]
        year = placement["year"]
        scope = _scope(_text(record.get("Current Level")), level, _text(record.get("Assigned Pathway")))
    else:
        level = PLACEMENT_MISSING
        year = ""
        scope = SCOPE_CURRENT
    decision = STATUS_CONDITIONAL if conditional else STATUS_REMAINING
    reason = _remaining_reason(status) if status in {HISTORICAL_FAILED, HISTORICAL_WITHDRAWN} else REASON_ELECTIVE
    return {
        "Student Code": student,
        "Pathway Readiness": record.get("Pathway Readiness", ""),
        "Assigned Pathway": record.get("Assigned Pathway", ""),
        "Possible Pathways": record.get("Possible Pathways", ""),
        "Current Level": record.get("Current Level", ""),
        "Course Code": code,
        "Course Title": meta.get("title") or code,
        "Study Plan Level": level,
        "Study Plan Year": year,
        "Course Type": "Elective",
        "Requirement Type": "Specialization",
        "Is Core": "No",
        "Is Elective": "Yes",
        "Elective Pool": pool,
        "Requirement Scope": scope,
        "Historical Status": status,
        "Remaining Status": decision,
        "Remaining Reason": reason,
        "Conditional Pathway": conditional or pd.NA,
        "Credit Hours": meta.get("credits", pd.NA),
    }


def identify_remaining_core_courses(requirement: dict[str, object], status: str) -> str:
    """Return the remaining reason for a core that has not been passed."""
    if status == HISTORICAL_PASSED:
        return REASON_SATISFIED
    return _remaining_reason(status)


def _summary_row(record: Mapping[str, object], cores: list[dict[str, object]], electives: dict[str, object], named_pending: int = 0) -> dict[str, object]:
    definite = [item for item in cores if item["decision"] == STATUS_REMAINING]
    conditional = [item for item in cores if item["decision"] == STATUS_CONDITIONAL]
    completed = [item for item in cores if item["decision"] == STATUS_COMPLETED and not item["conditional"]]
    failed = sum(item["status"] == HISTORICAL_FAILED for item in definite)
    withdrawn = sum(item["status"] == HISTORICAL_WITHDRAWN for item in definite)
    not_taken = sum(item["status"] == HISTORICAL_NOT_TAKEN for item in definite)
    common_remaining = len(definite)
    status = "No Definite Remaining Requirements" if not definite and electives["pending"] == 0 else "Definite Requirements Remaining"
    return {
        "Student Code": record["Student Code"],
        "Pathway Readiness": record["Pathway Readiness"],
        "Assigned Pathway": record["Assigned Pathway"],
        "Possible Pathways": record["Possible Pathways"],
        "Current Level": record["Current Level"],
        "Completed Core Count": len(completed),
        "Remaining Core Count": len(definite),
        "Failed Required Count": failed,
        "Withdrawn Required Count": withdrawn,
        "Not Yet Taken Core Count": not_taken,
        "Completed Elective Count": electives["completed"],
        "Pending Elective Slots": electives["pending"],
        "Common Remaining Count": common_remaining,
        "Conditional Remaining Count": len(conditional),
        "Remaining Course Status": status,
        "Pending Named Elective Slots": named_pending,
    }


def _unknown_summary(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "Student Code": record["Student Code"],
        "Pathway Readiness": READINESS_UNKNOWN,
        "Assigned Pathway": record["Assigned Pathway"],
        "Possible Pathways": record["Possible Pathways"],
        "Current Level": record["Current Level"],
        "Completed Core Count": 0,
        "Remaining Core Count": 0,
        "Failed Required Count": 0,
        "Withdrawn Required Count": 0,
        "Not Yet Taken Core Count": 0,
        "Completed Elective Count": 0,
        "Pending Elective Slots": 0,
        "Common Remaining Count": 0,
        "Conditional Remaining Count": 0,
        "Remaining Course Status": STATUS_PATHWAY_REQUIRED,
        "Pending Named Elective Slots": 0,
    }


def _unknown_remaining_row(record: Mapping[str, object]) -> dict[str, object]:
    row = {column: pd.NA for column in REMAINING_COLUMNS}
    row.update({
        "Student Code": record["Student Code"],
        "Pathway Readiness": READINESS_UNKNOWN,
        "Assigned Pathway": record["Assigned Pathway"],
        "Possible Pathways": record["Possible Pathways"],
        "Current Level": record["Current Level"],
        "Remaining Status": STATUS_PATHWAY_REQUIRED,
        "Remaining Reason": REASON_UNKNOWN,
        "Requirement Scope": pd.NA,
    })
    return row


def _stamp(items: list[dict[str, object]], record: Mapping[str, object]) -> list[dict[str, object]]:
    for item in items:
        item["readiness"] = record["Pathway Readiness"]
        item["assigned"] = record["Assigned Pathway"]
        item["possible"] = record["Possible Pathways"]
        item["level_name"] = record["Current Level"]
    return items


def _course_record(record: Mapping[str, object], *, core: bool) -> dict[str, object]:
    return {
        "code": str(record["Course Code"]),
        "title": _text(record.get("Course Title")),
        "level": _text(record.get("Level")),
        "year": _text(record.get("Year")),
        "course_type": _text(record.get("Course Type")),
        "requirement_type": _text(record.get("Requirement Type")),
        "pool": _text(record.get("Elective Pool")),
        "source": _text(record.get("Specialization Path")),
        "core": core,
    }


def _merge_common(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    merged = dict(left)
    merged["conditional"] = ""
    merged["scope"] = SCOPE_COMMON if left["scope"] != SCOPE_FUTURE and right["scope"] != SCOPE_FUTURE else SCOPE_FUTURE
    for field in ("level", "year"):
        if left[field] != right[field]:
            merged[field] = f"{left[field]} | {right[field]}"
    return merged


def _slot_groups(slots: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for slot in slots:
        groups.setdefault((_text(slot.get("pool")), slot["scope"]), []).append(slot)
    return groups


def _scope(student_level: str, plan_level: str, pathway: str) -> str:
    if student_level not in LEVEL_RANK:
        if pathway == PATH_COMMON:
            return SCOPE_CURRENT
        raise RuntimeError(f"A {pathway} student has no official level.")
    if LEVEL_RANK[plan_level] <= LEVEL_RANK[student_level]:
        return SCOPE_CURRENT
    return SCOPE_FUTURE


def _final_status(attempts: list[dict[str, object]]) -> str:
    if any(item["passed"] for item in attempts):
        return HISTORICAL_PASSED
    if any(item["failed"] for item in attempts):
        return HISTORICAL_FAILED
    if any(item["withdrawn"] for item in attempts):
        return HISTORICAL_WITHDRAWN
    return HISTORICAL_NOT_TAKEN


def _remaining_reason(status: str) -> str:
    if status == HISTORICAL_FAILED:
        return REASON_FAILED
    if status == HISTORICAL_WITHDRAWN:
        return REASON_WITHDRAWN
    return REASON_NOT_TAKEN


def _attempt_label(record: Mapping[str, object]) -> str:
    if _flag(record.get("Is Passed")):
        label = HISTORICAL_PASSED
    elif _flag(record.get("Is Failed")):
        label = HISTORICAL_FAILED
    elif _flag(record.get("Is Withdrawn")):
        label = HISTORICAL_WITHDRAWN
    else:
        label = "Other"
    if _flag(record.get("Is Repeated For Improvement")) or _text(record.get("Remarks")).upper() == "N":
        label += "; Repeated to improve GPA"
    return label


def _plan_codes(requirements: dict[str, object]) -> set[str]:
    codes = set()
    for rows in requirements["cores"].values():
        codes.update(item["code"] for item in rows)
    for courses in requirements["pools"].values():
        codes.update(courses)
    return codes


def _completed_elective_text(electives: pd.DataFrame) -> str:
    if electives.empty:
        return "None"
    parts = []
    for record in electives.to_dict(orient="records"):
        codes = _text(record.get("Completed Elective Codes"))
        if codes:
            parts.append(f"{record['Elective Pool']}: {codes}")
    return "; ".join(parts) if parts else "None"


def _pending_elective_text(electives: pd.DataFrame) -> str:
    if electives.empty:
        return "None"
    parts = [f"{record['Elective Pool']} ({int(record['Pending Slots'])})" for record in electives.to_dict(orient="records") if int(record["Pending Slots"]) > 0]
    return "; ".join(parts) if parts else "None"


def _conditional_text(courses: pd.DataFrame, electives: pd.DataFrame) -> str:
    parts = []
    if not courses.empty:
        for pathway, group in courses.groupby("Conditional Pathway", sort=False):
            parts.append(f"If {pathway}: {'; '.join(group['Course Code'].astype(str))}")
    if not electives.empty:
        conditional = electives.loc[electives["Conditional"].eq("Yes") & electives["Pending Slots"].gt(0)]
        for record in conditional.to_dict(orient="records"):
            parts.append(f"If {record['Pathway']}: {record['Elective Pool']} x{int(record['Pending Slots'])}")
    return " | ".join(parts) if parts else "None"


def _join(values: pd.Series) -> str:
    codes = [str(value) for value in values if _text(value)]
    return "; ".join(codes) if codes else "None"


def _frame(rows: list[dict[str, object]], columns: list[str], sort_by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    existing = [column for column in sort_by if column in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True)


def _check(check: str, result: object, *, fail: bool = False) -> dict[str, object]:
    return {"Check": check, "Result": int(result), "Status": "FAIL" if fail else "PASS"}


def _yes(value: object) -> bool:
    return _text(value).casefold() == "yes"


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if _missing(value):
        return False
    return str(value).strip().casefold() in {"true", "yes", "y", "1"}


def _text(value: object) -> str:
    if _missing(value):
        return ""
    return str(value).strip()


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _year_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _term_value(value: object) -> int:
    return 0 if "spring" in _text(value).casefold() else 1


def _term_name(value: object) -> str:
    text = _text(value)
    if "spring" in text.casefold():
        return "Spring"
    if "fall" in text.casefold():
        return "Fall"
    return text


def _attempt_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _format(worksheet, widths: dict[str, int], *, integer_columns: set[str] | None = None, wrap_columns: set[str] | None = None) -> None:
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
    integer_columns = integer_columns or set()
    wrap_columns = wrap_columns or set()
    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
        for cell in row:
            letter = get_column_letter(cell.column)
            if letter in integer_columns and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = "0"
            if letter in wrap_columns:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
