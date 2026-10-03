"""Chronological repeat types derived from the previous attempt.

Stored repeat flags stay on the source row for audit. They do not decide the type.
"""

from __future__ import annotations

import pandas as pd

REPEAT_FAILURE = "Repeat Due To Failure"
REPEAT_WITHDRAWAL = "Repeat After Withdrawal"
REPEAT_IMPROVEMENT = "Repeat For Improvement"
REPEAT_UNRESOLVED = "Repeat Type Unresolved"
OUTCOME_FAILED_RECOVERED = "Failed — recovered from independent validated fields"
OUTCOME_POSTPONED = "Officially Postponed"
OUTCOME_MANUAL = "Manual Review"
OUTCOME_PASSED = "Passed"
OUTCOME_FAILED = "Failed"
OUTCOME_WITHDRAWN = "Withdrawn"
FAILING_GRADES = {"F", "FW"}


def derive_repeat_classification(transcript: pd.DataFrame) -> pd.DataFrame:
    """Return one row per transcript attempt with a derived repeat type.

    The first attempt of a student-course has no derived repeat type.
    Later attempts are classified from the immediately previous attempt.
    """

    if transcript is None or transcript.empty:
        return pd.DataFrame(columns=_columns())
    frame = transcript.copy()
    source_index = frame.index.to_numpy()
    frame["_row"] = range(len(frame))
    frame["_student"] = frame["Student Code"].map(_text) if "Student Code" in frame.columns else ""
    frame["_code"] = frame["Course Code"].map(_text) if "Course Code" in frame.columns else ""
    frame["_year"] = frame.apply(_year_key, axis=1) if len(frame) else []
    frame["_term"] = frame["Semester"].map(_term_key) if "Semester" in frame.columns else 9
    frame["_attempt"] = frame["Attempt Number"].map(_attempt) if "Attempt Number" in frame.columns else 1
    frame = frame.sort_values(["_student", "_code", "_year", "_term", "_attempt", "_row"], kind="mergesort")
    derived: list[str] = []
    previous_outcome: list[str] = []
    evidence: list[str] = []
    resolution: list[str] = []
    outcome_evidence: list[str] = []
    confidence: list[str] = []
    manual: list[str] = []
    mismatch: list[str] = []
    outcome: list[str] = []
    last: dict[tuple[str, str], dict[str, object]] = {}
    for record in frame.to_dict(orient="records"):
        key = (str(record["_student"]), str(record["_code"]))
        current = _attempt_outcome(record)
        outcome.append(current["label"])
        outcome_evidence.append(str(current["evidence"]))
        confidence.append(str(current["confidence"]))
        manual.append(str(current["manual_review"]))
        prior = last.get(key)
        repeat_type = ""
        repeat_evidence = ""
        repeat_resolution = ""
        if prior is not None and int(record["_attempt"]) > 1:
            repeat_type = _type_from_previous(prior)
            repeat_evidence = (
                f"Previous attempt outcome: {prior['label']}. "
                f"Grade: {prior['grade'] or 'blank'}. Result: {prior['result'] or 'blank'}."
            )
            repeat_resolution = "Unresolved" if repeat_type == REPEAT_UNRESOLVED else "Resolved"
        derived.append(repeat_type)
        previous_outcome.append("" if prior is None else str(prior["label"]))
        evidence.append(repeat_evidence)
        resolution.append(repeat_resolution)
        mismatch.append(_mismatch(record, repeat_type))
        last[key] = current
    frame["Derived Repeat Type"] = derived
    frame["Previous Attempt Outcome"] = previous_outcome
    frame["Repeat Evidence"] = evidence
    frame["Repeat Resolution Status"] = resolution
    frame["Derived Academic Outcome"] = outcome
    frame["Derived Outcome Evidence"] = outcome_evidence
    frame["Outcome Confidence"] = confidence
    frame["Outcome Manual Review"] = manual
    frame["Stored Failure Repeat Flag"] = frame["Is Repeated Due To Failure"].map(_flag) if "Is Repeated Due To Failure" in frame.columns else False
    frame["Stored Improvement Repeat Flag"] = frame["Is Repeated For Improvement"].map(_flag) if "Is Repeated For Improvement" in frame.columns else False
    frame["Repeat Flag Mismatch"] = mismatch
    frame = frame.sort_values("_row", kind="mergesort")
    frame.index = source_index[frame["_row"].to_numpy()]
    return frame.drop(columns=["_row", "_student", "_code", "_year", "_term", "_attempt"])


def _type_from_previous(previous: dict[str, object]) -> str:
    kind = previous["kind"]
    if kind == "withdrawn":
        return REPEAT_WITHDRAWAL
    if kind == "failed":
        return REPEAT_FAILURE
    if kind == "passed" and previous["remarks"] == "N":
        return REPEAT_IMPROVEMENT
    return REPEAT_UNRESOLVED


def _attempt_outcome(record: dict[str, object]) -> dict[str, object]:
    grade = _text(record.get("Grade")).upper()
    result = _text(record.get("Result")).upper()
    remarks = _text(record.get("Remarks")).upper()
    withdrawn = _flag(record.get("Is Withdrawn")) or grade == "W" or result == "W"
    failed = _flag(record.get("Is Failed")) or result == "F" or grade in FAILING_GRADES
    passed = _flag(record.get("Is Passed")) or result == "P"
    raw_result_valid = result in {"P", "F", "W"}
    grade_point = _text(record.get("Grade Point"))
    if grade == "OP":
        label = OUTCOME_POSTPONED
        kind = "unavailable"
        outcome_evidence = "Grade = OP"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif withdrawn:
        label = OUTCOME_WITHDRAWN
        kind = "withdrawn"
        outcome_evidence = "Withdrawal flag, grade, or result"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif failed and raw_result_valid:
        label = OUTCOME_FAILED
        kind = "failed"
        outcome_evidence = "Grade, result, or Is Failed"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif failed and not raw_result_valid and _flag(record.get("Is Failed")) and grade in FAILING_GRADES:
        label = OUTCOME_FAILED_RECOVERED
        kind = "failed"
        outcome_evidence = "Grade + Grade Point + Is Failed"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif failed:
        label = OUTCOME_FAILED
        kind = "failed"
        outcome_evidence = "Grade, result, or Is Failed"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif passed:
        label = OUTCOME_PASSED
        kind = "passed"
        outcome_evidence = "Is Passed or Result = P"
        outcome_confidence = "Deterministic"
        outcome_manual = "No"
    elif grade == "" and result == "":
        label = OUTCOME_MANUAL
        kind = "unavailable"
        outcome_evidence = "Grade and Result are blank"
        outcome_confidence = "Unresolved"
        outcome_manual = "Yes"
    else:
        label = OUTCOME_MANUAL
        kind = "unavailable"
        outcome_evidence = "Grade, Result, and status flags do not determine one outcome"
        outcome_confidence = "Unresolved"
        outcome_manual = "Yes"
    return {
        "kind": kind,
        "remarks": remarks,
        "label": label,
        "grade": grade,
        "result": result,
        "grade_point": grade_point,
        "evidence": outcome_evidence,
        "confidence": outcome_confidence,
        "manual_review": outcome_manual,
    }


def _mismatch(record: dict[str, object], repeat_type: str) -> str:
    if repeat_type == "":
        return ""
    stored_failure = _flag(record.get("Is Repeated Due To Failure"))
    stored_improvement = _flag(record.get("Is Repeated For Improvement"))
    notes: list[str] = []
    if repeat_type == REPEAT_FAILURE and not stored_failure:
        notes.append("Derived failure repeat is not marked on the stored failure flag.")
    if repeat_type == REPEAT_IMPROVEMENT and not stored_improvement:
        notes.append("Derived improvement repeat is not marked on the stored improvement flag.")
    if repeat_type != REPEAT_FAILURE and stored_failure:
        notes.append("Stored failure flag disagrees with the derived repeat type.")
    if repeat_type != REPEAT_IMPROVEMENT and stored_improvement:
        notes.append("Stored improvement flag disagrees with the derived repeat type.")
    return " ".join(notes)


def _columns() -> list[str]:
    return [
        "Derived Repeat Type",
        "Previous Attempt Outcome",
        "Repeat Evidence",
        "Repeat Resolution Status",
        "Derived Academic Outcome",
        "Derived Outcome Evidence",
        "Outcome Confidence",
        "Outcome Manual Review",
        "Stored Failure Repeat Flag",
        "Stored Improvement Repeat Flag",
        "Repeat Flag Mismatch",
    ]


def _year_key(record: pd.Series) -> int:
    year = _attempt(record.get("Academic Year")) if "Academic Year" in record else 0
    if year:
        return year
    text = _text(record.get("Semester"))
    digits = "".join(character for character in text if character.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else 0


def _term_key(value: object) -> int:
    text = _text(value).casefold()
    if "spring" in text:
        return 1
    if "summer" in text:
        return 2
    if "fall" in text or "autumn" in text:
        return 3
    return 9


def _attempt(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip().casefold() in {"true", "yes", "y", "1"}


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()
