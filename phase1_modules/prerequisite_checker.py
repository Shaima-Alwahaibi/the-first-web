"""Check prerequisite eligibility for remaining courses.

The checker reads official prerequisite text and Fall 2025 attempt history.
It does not assign pathways, recalculate remaining courses, or apply course-load limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from preparation_checks import normalize_course_code, write_atomic
from remaining_courses import FUTURE_CONTINUATION, PATHWAY_BLOCKS
from study_plan_matching import (
    HOLDOUT_TOKEN,
    PATH_DSAI,
    PATH_SE,
    READINESS_FULL,
    READINESS_PARTIAL,
    READINESS_UNKNOWN,
)

ELIGIBLE = "Eligible"
NOT_ELIGIBLE = "Not Eligible"
ELIGIBLE_ADVISOR = "Eligible with Advisor Approval"
PATHWAY_REQUIRED = "Pathway Resolution Required"
EVIDENCE_AVAILABLE = "Historical Evidence Available"
EVIDENCE_INSUFFICIENT = "Insufficient Historical Evidence"
REVIEW_YES = "Yes"
REVIEW_NO = "No"
STATUS_REMAINING = "Remaining"
STATUS_CONDITIONAL = "Conditional"

TYPE_NONE = "No prerequisite"
TYPE_SINGLE = "Single"
TYPE_AND = "AND"
TYPE_OR = "OR"
TYPE_COMBINED = "Combined"

SCOPE_SHARED = "Shared Rule"
SCOPE_PATHWAY = "Pathway-Specific Rule"
SCOPE_PATHWAY_SAFE = "Pathway-Safe SE/DSAI Elective"
SCOPE_PATHWAY_SAFE_CANDIDATE = "Pathway-Safe Candidate"
SCOPE_CONDITIONAL_SE = "Conditional SE Candidate"
SCOPE_CONDITIONAL_DSAI = "Conditional DSAI Candidate"
SCOPE_PATHWAY_DEPENDENT = "Pathway-Dependent Eligibility"
SCOPE_NONE = "No Prerequisite"

NONE_TOKENS = {
    "",
    "NONE",
    "N/A",
    "NA",
    "NULL",
    "NIL",
    "-",
    "NO PREREQUISITE",
    "NO PREREQUISITE REQUIRED",
}
COURSE_PATTERN = re.compile(r"[A-Z]{2,}[0-9]{3,}[A-Z0-9]*")
TOKEN_PATTERN = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bCO-REQUISITE\b|\bCOREQUISITE\b|[A-Z]{2,}[0-9]{3,}[A-Z0-9]*")
SHEET_TYPE_TO_PARSED = {
    "SINGLE": TYPE_SINGLE,
    "AND": TYPE_AND,
    "OR": TYPE_OR,
    "COMBINED": TYPE_COMBINED,
}

RESULT_COLUMNS = [
    "Student Code",
    "Pathway Readiness",
    "Assigned Pathway",
    "Possible Pathways",
    "Conditional Pathway",
    "Course Code",
    "Course Title",
    "Requirement Scope",
    "Remaining Status",
    "Prerequisite Rule Original",
    "Prerequisite Rule Normalized",
    "Co-requisite Rule",
    "Rule Type",
    "Prerequisite Scope",
    "Eligibility Status",
    "Eligibility Evidence Status",
    "Manual Review Required",
    "Satisfied Prerequisites",
    "Missing Prerequisites",
    "Advisor Approval Reason",
    "Eligibility Reason",
    "Candidate Scope",
]
RULE_COLUMNS = [
    "Course Code",
    "Course Title",
    "Specialization Path",
    "Level",
    "Original Rule",
    "Normalized Rule",
    "Parsed Expression",
    "Referenced Course Codes",
    "Rule Type",
    "Parse Status",
    "Parse Notes",
]
AUDIT_COLUMNS = [
    "Student Code",
    "Course Code",
    "Referenced Prerequisite",
    "Historical Attempts",
    "Passed Before Holdout",
    "Evaluation Result",
]


class PrerequisiteParseError(ValueError):
    """A prerequisite expression cannot be parsed without changing its meaning."""


@dataclass(frozen=True)
class Expression:
    """A course code or an AND/OR node. Children stay in source order."""

    kind: str
    code: str = ""
    items: tuple["Expression", ...] = ()


@dataclass(frozen=True)
class ConcurrentException:
    """An official permission to study a failed prerequisite with the target course."""

    course_code: str
    prerequisite_code: str
    source: str


@dataclass(frozen=True)
class Judgment:
    """One expression evaluated against a student's passed courses."""

    satisfied: bool
    satisfied_codes: tuple[str, ...]
    missing_codes: tuple[str, ...]
    unknown_codes: tuple[str, ...]
    reason: str
    unavailable_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedRule:
    """One official prerequisite row, with the study-plan placements that use it."""

    course_code: str
    course_title: str
    original: str
    normalized: str
    expression: Expression | None
    co_requisite: Expression | None
    co_requisite_text: str
    rule_type: str
    referenced: tuple[str, ...]
    placements: tuple[tuple[str, str, str], ...]
    parse_status: str
    parse_notes: str
    sheet_index: int


def inspect_prerequisite_schema(rules: pd.DataFrame, advising: pd.DataFrame) -> dict[str, object]:
    """Return the official column names, row count, and stored rule-type counts."""

    _require_columns(rules, ["Course Code", "Course Title", "Prerequisite Rule", "Rule Type"], "Prerequisite Rules")
    _require_columns(advising, ["Rule Area", "Rule", "Source/Notes"], "Advising Rules")
    return {
        "prerequisite_rows": int(len(rules)),
        "prerequisite_columns": list(rules.columns),
        "rule_type_counts": rules["Rule Type"].value_counts(dropna=False).astype(int).to_dict(),
        "blank_prerequisite_rules": int(rules["Prerequisite Rule"].isna().sum()),
        "advising_rows": int(len(advising)),
        "advising_areas": advising["Rule Area"].astype(str).tolist(),
        "co_requisite_column": "Co-requisite" in rules.columns,
    }


def normalize_rule_text(value: object) -> str:
    """Collapse whitespace and recognize an explicit empty prerequisite."""

    text = _text(value).upper().replace("–", "-").replace("—", "-")
    text = text.replace("COREQUISITE", "CO-REQUISITE")
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text in NONE_TOKENS else text


def tokenize_rule(text: str) -> list[str]:
    """Split a normalized rule into course codes, operators, and parentheses."""

    if text == "":
        return []
    tokens = TOKEN_PATTERN.findall(text)
    consumed = TOKEN_PATTERN.sub("", text)
    if consumed.strip():
        raise PrerequisiteParseError(f"Unrecognized prerequisite text: {text}")
    if not tokens:
        raise PrerequisiteParseError(f"Prerequisite text has no course code or operator: {text}")
    return tokens


def parse_rule(value: object) -> tuple[Expression | None, Expression | None, str]:
    """Parse prerequisite text. Return the prerequisite, co-requisite, and normalized text.

    AND binds more tightly than OR. Parentheses override that order. The official
    combined rules already contain parentheses, so the precedence does not change them.
    """

    normalized = normalize_rule_text(value)
    if normalized == "":
        return None, None, ""
    prerequisite_text, co_text = _split_corequisite(normalized)
    prerequisite = _parse_tokens(tokenize_rule(prerequisite_text)) if prerequisite_text else None
    co_requisite = _parse_tokens(tokenize_rule(co_text)) if co_text else None
    normalized_parts = []
    if prerequisite is not None:
        normalized_parts.append(format_expression(prerequisite))
    if co_requisite is not None:
        normalized_parts.append("CO-REQUISITE " + format_expression(co_requisite))
    return prerequisite, co_requisite, " | ".join(normalized_parts)


def format_expression(expression: Expression | None) -> str:
    """Render an expression with parentheses only where the source grouping needs them."""

    if expression is None:
        return ""
    return _format_node(expression, parent="")


def referenced_codes(expression: Expression | None) -> tuple[str, ...]:
    """List course codes in source order, without duplicates."""

    if expression is None:
        return ()
    found: list[str] = []
    _collect_codes(expression, found)
    return tuple(dict.fromkeys(found))


def classify_rule(expression: Expression | None) -> str:
    """Classify a parsed prerequisite as none, single, AND, OR, or combined."""

    if expression is None:
        return TYPE_NONE
    operators: set[str] = set()
    _collect_operators(expression, operators)
    if not operators:
        return TYPE_SINGLE
    if operators == {"and"}:
        return TYPE_AND
    if operators == {"or"}:
        return TYPE_OR
    return TYPE_COMBINED


def find_concurrent_exceptions(advising: pd.DataFrame) -> list[ConcurrentException]:
    """Read course-specific concurrent exceptions from Advising Rules.

    A row is used only when it names advisor approval, concurrent or joint study,
    and at least one target course with one prerequisite course. The current
    official sheet has no such row.
    """

    _require_columns(advising, ["Rule Area", "Rule"], "Advising Rules")
    exceptions: list[ConcurrentException] = []
    for record in advising.to_dict(orient="records"):
        text = normalize_rule_text(record.get("Rule"))
        if not _is_concurrent_approval_text(text):
            continue
        codes = COURSE_PATTERN.findall(text)
        unique = list(dict.fromkeys(codes))
        if len(unique) < 2:
            continue
        source = _text(record.get("Rule Area")) or "Advising Rules"
        for prerequisite in unique[1:]:
            exceptions.append(ConcurrentException(unique[0], prerequisite, source))
    return exceptions


def build_passed_course_index(transcript: pd.DataFrame) -> dict[str, object]:
    """Index Fall 2025 attempts. A later pass satisfies the prerequisite."""

    required = ["Student Code", "Course Code", "Semester", "Academic Year", "Attempt Number", "Is Passed", "Is Failed", "Is Withdrawn"]
    _require_columns(transcript, required, "historical transcript")
    holdout = transcript["Semester"].astype(str).str.contains(HOLDOUT_TOKEN) | transcript["Academic Year"].astype(str).str.contains(HOLDOUT_TOKEN)
    history = transcript.loc[~holdout].copy()
    students: dict[str, dict[str, list[dict[str, object]]]] = {}
    for record in history.to_dict(orient="records"):
        student = _text(record.get("Student Code"))
        code = normalize_course_code(record.get("Course Code"))
        if student == "" or _missing(code):
            continue
        course_code = str(code)
        attempt = {
            "year": _year_value(record.get("Academic Year")),
            "term": _term_value(record.get("Term", record.get("Semester"))),
            "attempt": _attempt_value(record.get("Attempt Number")),
            "passed": _flag(record.get("Is Passed")),
            "failed": _flag(record.get("Is Failed")),
            "withdrawn": _flag(record.get("Is Withdrawn")),
            "label": _attempt_label(record),
        }
        students.setdefault(student, {}).setdefault(course_code, []).append(attempt)
    for courses in students.values():
        for attempts in courses.values():
            attempts.sort(key=lambda item: (item["year"], item["term"], item["attempt"]))
    return {"students": students, "spring_2026_leakage": int(holdout.sum())}


def evaluate_prerequisite(
    rule_text: object,
    passed: set[str],
    failed: set[str] | None = None,
    *,
    co_requisite: object = "",
    concurrent_corequisite: bool = False,
    concurrent_exceptions: list[ConcurrentException] | None = None,
    course_code: str = "",
    known_codes: set[str] | None = None,
    attempted: set[str] | None = None,
) -> dict[str, object]:
    """Evaluate one rule. Blank text means no prerequisite."""

    expression, parsed_corequisite, normalized = parse_rule(rule_text)
    extra_corequisite, _, _ = parse_rule(co_requisite)
    corequisite = extra_corequisite or parsed_corequisite
    failed_codes = failed or set()
    attempted_codes = attempted or set()
    exceptions = concurrent_exceptions or []
    unknown = _unknown_codes(expression, corequisite, known_codes)
    judgment = (
        _judge(expression, passed, unknown, failed_codes, attempted_codes)
        if expression is not None
        else Judgment(True, (), (), (), "No prerequisite required")
    )
    status = ELIGIBLE if judgment.satisfied else NOT_ELIGIBLE
    reason = judgment.reason
    advisor_reason = ""
    satisfied = list(judgment.satisfied_codes)
    missing = list(judgment.missing_codes)
    if expression is not None and not judgment.satisfied:
        covered = _exception_codes(course_code, exceptions, failed_codes)
        if covered:
            provisional = set(passed) | covered
            retry = _judge(expression, provisional, unknown, failed_codes, attempted_codes)
            if retry.satisfied:
                status = ELIGIBLE_ADVISOR
                satisfied = list(retry.satisfied_codes)
                missing = []
                used = sorted(covered & set(retry.satisfied_codes))
                advisor_reason = "Official rule allows " + course_code + " and " + ", ".join(used) + " to be studied together after the prerequisite was failed."
                reason = advisor_reason
    if corequisite is not None:
        co_judgment = _judge(corequisite, passed, unknown, failed_codes, attempted_codes)
        co_text = format_expression(corequisite)
        if co_judgment.satisfied:
            reason = _join_reason(reason, f"Co-requisite {co_text} is already completed.")
            satisfied = list(dict.fromkeys([*satisfied, *co_judgment.satisfied_codes]))
        elif concurrent_corequisite and status in {ELIGIBLE, ELIGIBLE_ADVISOR}:
            status = ELIGIBLE_ADVISOR
            advisor_reason = _join_reason(advisor_reason, f"Co-requisite {co_text} is not completed. The official rule permits concurrent enrollment with advisor approval.")
            reason = advisor_reason
        else:
            status = NOT_ELIGIBLE
            missing = list(dict.fromkeys([*missing, *co_judgment.missing_codes]))
            reason = _join_reason(reason, f"Missing co-requisite: {co_text}.")
            satisfied = [code for code in satisfied if code not in set(co_judgment.missing_codes)]
            missing = [code for code in missing if code not in set(co_judgment.unavailable_codes)]
    unavailable = list(judgment.unavailable_codes)
    if corequisite is not None:
        unavailable = list(dict.fromkeys([*unavailable, *co_judgment.unavailable_codes]))
    if status in {ELIGIBLE, ELIGIBLE_ADVISOR}:
        unavailable = []
    confirmed_missing = list(missing)
    if status == NOT_ELIGIBLE:
        missing = list(dict.fromkeys([*confirmed_missing, *unavailable]))
    evidence_status, manual_review = _evidence_fields(status, confirmed_missing, unavailable)
    return {
        "status": status,
        "evidence_status": evidence_status,
        "manual_review": manual_review,
        "unavailable": unavailable,
        "normalized": normalized if co_requisite == "" else _join_rule(normalized, format_expression(corequisite)),
        "co_requisite": format_expression(corequisite),
        "rule_type": classify_rule(expression),
        "satisfied": satisfied,
        "missing": missing,
        "unknown": sorted(set(judgment.unknown_codes) | set(_unknown_codes(corequisite, None, known_codes))),
        "advisor_reason": advisor_reason,
        "reason": reason,
        "referenced": list(dict.fromkeys([*referenced_codes(expression), *referenced_codes(corequisite)])),
    }


def build_prerequisite_output(
    remaining: pd.DataFrame,
    transcript: pd.DataFrame,
    plan: pd.DataFrame,
    rules: pd.DataFrame,
    pools: pd.DataFrame,
    advising: pd.DataFrame,
    validation_lists: pd.DataFrame,
    profiles: pd.DataFrame,
    original_transcript: pd.DataFrame | None = None,
    excluded_history: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate remaining and conditional rows. Pathway-unknown students get no course plan.

    Foundation course rows recovered from the original transcript or the exclusion
    sheet are used only as prerequisite evidence. They are not added to remaining courses.
    """

    catalog = build_rule_catalog(rules, plan, pools)
    history = build_passed_course_index(transcript)
    _copy_foundation_attempts(history, original_transcript)
    _copy_foundation_attempts(history, excluded_history)
    known = _known_codes(plan, pools, rules, transcript, advising, validation_lists)
    exceptions = find_concurrent_exceptions(advising)
    foundation_codes = sorted({code for rule in catalog["rules"] for code in rule.referenced if str(code).startswith("FP")})
    results: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    unknown_references: set[str] = set()
    for record in remaining.to_dict(orient="records"):
        readiness = _text(record.get("Pathway Readiness"))
        course = normalize_course_code(record.get("Course Code"))
        if readiness == READINESS_UNKNOWN or _missing(course):
            results.append(_pathway_row(record))
            continue
        selected = _select_rules(catalog["by_code"].get(str(course), []), record, pools)
        if not selected:
            results.append(_missing_rule_row(record))
            continue
        evaluation = _evaluate_selected(str(course), selected, record, history, known, exceptions)
        unknown_references.update(evaluation["unknown"])
        results.append(_result_row(record, selected, evaluation))
        audits.extend(_audit_rows(record, evaluation, history))
    output = {
        "results": _frame(results, RESULT_COLUMNS, ["Student Code", "Course Code", "Conditional Pathway"]),
        "rule_audit": catalog["audit"],
        "student_audit": _frame(audits, AUDIT_COLUMNS, ["Student Code", "Course Code", "Referenced Prerequisite"]),
        "pool_mismatches": catalog["pool_mismatches"],
        "exceptions": exceptions,
        "foundation_codes": foundation_codes,
        "foundation_attempts": {
            "original": _foundation_attempt_count(original_transcript, foundation_codes),
            "excluded": _foundation_attempt_count(excluded_history, foundation_codes),
        },
        "spring_2026_leakage": history["spring_2026_leakage"],
        "profiles": profiles,
        "unknown_references": sorted(unknown_references),
    }
    output["validation"] = validate_prerequisite_results(output, remaining)
    return output


def build_rule_catalog(rules: pd.DataFrame, plan: pd.DataFrame, pools: pd.DataFrame) -> dict[str, object]:
    """Parse every Prerequisite Rules row and attach the study-plan placement."""

    _require_columns(rules, ["Course Code", "Course Title", "Prerequisite Rule", "Rule Type"], "Prerequisite Rules")
    _require_columns(plan, ["Specialization Path", "Level", "Year", "Course Code", "Prerequisite Rule"], "Study Plan Courses")
    parsed: list[ParsedRule] = []
    audit_rows: list[dict[str, object]] = []
    for index, record in enumerate(rules.to_dict(orient="records")):
        code = normalize_course_code(record.get("Course Code"))
        original = _text(record.get("Prerequisite Rule"))
        notes: list[str] = []
        try:
            expression, co_requisite, normalized = parse_rule(original)
            status = "OK"
            rule_type = classify_rule(expression)
            sheet_type = _text(record.get("Rule Type")).upper()
            expected = SHEET_TYPE_TO_PARSED.get(sheet_type)
            if original == "" and sheet_type == "":
                notes.append("The Prerequisite Rules sheet lists this course with an empty rule, so there is no prerequisite.")
            elif expected and expected != rule_type:
                notes.append(f"Sheet rule type {sheet_type} does not match the parsed type {rule_type}.")
            if co_requisite is not None:
                notes.append("The rule text contains a co-requisite clause.")
        except PrerequisiteParseError as exc:
            expression = None
            co_requisite = None
            normalized = normalize_rule_text(original)
            status = "Failed"
            rule_type = ""
            notes.append(str(exc))
        course_code = "" if _missing(code) else str(code)
        placements = _placements_for(plan, course_code, original)
        if not placements and course_code:
            notes.append("Listed on Prerequisite Rules and not as a Study Plan Courses core row.")
        item = ParsedRule(
            course_code,
            _text(record.get("Course Title")),
            original,
            normalized,
            expression,
            co_requisite,
            format_expression(co_requisite),
            rule_type,
            tuple(dict.fromkeys([*referenced_codes(expression), *referenced_codes(co_requisite)])),
            placements,
            status,
            " ".join(notes),
            index,
        )
        parsed.append(item)
        audit_rows.append(_rule_audit_row(item))
    by_code: dict[str, list[ParsedRule]] = {}
    for item in parsed:
        by_code.setdefault(item.course_code, []).append(item)
    return {
        "rules": parsed,
        "by_code": by_code,
        "audit": _frame(audit_rows, RULE_COLUMNS, ["Course Code", "Specialization Path", "Original Rule"]),
        "pool_mismatches": _unmatched_pool_rules(pools, by_code),
    }


def validate_prerequisite_results(output: Mapping[str, object], remaining: pd.DataFrame) -> pd.DataFrame:
    """Return PASS, REVIEW, or FAIL checks. A parse failure on a used rule is FAIL."""

    results = output["results"]
    audit = output["rule_audit"]
    evaluated = results.loc[results["Course Code"].map(lambda value: _text(value) != "")]
    eligible = int(evaluated["Eligibility Status"].eq(ELIGIBLE).sum())
    blocked = int(evaluated["Eligibility Status"].eq(NOT_ELIGIBLE).sum())
    advisor = int(evaluated["Eligibility Status"].eq(ELIGIBLE_ADVISOR).sum())
    official = {ELIGIBLE, NOT_ELIGIBLE, ELIGIBLE_ADVISOR}
    unexpected = int((~evaluated["Eligibility Status"].isin(official)).sum()) if not evaluated.empty else 0
    parse_failures = int(audit["Parse Status"].eq("Failed").sum()) if not audit.empty else 0
    used_failures = _used_parse_failures(evaluated, audit)
    unknown = output["unknown_references"]
    missing_rules = int(results["Eligibility Reason"].eq("Missing source rule").sum())
    leakage = int(output["spring_2026_leakage"])
    duplicate_key = ["Student Code", "Course Code", "Conditional Pathway", "Remaining Status"]
    duplicates = int(evaluated.duplicated(duplicate_key).sum()) if not evaluated.empty else 0
    conditional_input = remaining.loc[remaining["Remaining Status"].eq(STATUS_CONDITIONAL)]
    conditional_output = results.loc[results["Remaining Status"].eq(STATUS_CONDITIONAL)]
    conditional_marked = len(conditional_output) == len(conditional_input) and (
        conditional_output.empty or bool(conditional_output["Remaining Status"].eq(STATUS_CONDITIONAL).all())
    )
    profile_ids = set(output["profiles"]["Student Code"].map(_text)) if "Student Code" in output["profiles"].columns else set()
    result_ids = set(results["Student Code"].map(_text))
    absent_profiles = sorted(profile_ids - result_ids)
    unknown_students = results.loc[results["Pathway Readiness"].eq(READINESS_UNKNOWN)]
    unknown_evaluated = int(unknown_students["Course Code"].map(lambda value: _text(value) != "").sum()) if not unknown_students.empty else 0
    type_counts = audit["Rule Type"].value_counts() if not audit.empty else pd.Series(dtype=int)
    exceptions = output["exceptions"]
    foundation = output["foundation_codes"]
    lacking = evaluated.loc[evaluated["Eligibility Evidence Status"].eq(EVIDENCE_INSUFFICIENT)] if not evaluated.empty else evaluated
    lacking_students = int(lacking["Student Code"].nunique()) if not lacking.empty else 0
    resolved = _resolved_foundation_rows(output["student_audit"], foundation)
    reason = evaluated["Eligibility Reason"].astype(str) if not evaluated.empty else pd.Series(dtype=str)
    incorrect_failure = 0 if evaluated.empty else int((
        (reason.str.contains("does not contain student-level evidence for that foundation course", case=False, na=False) & evaluated["Eligibility Evidence Status"].ne(EVIDENCE_INSUFFICIENT))
        | (evaluated["Eligibility Evidence Status"].eq(EVIDENCE_INSUFFICIENT) & evaluated["Manual Review Required"].ne(REVIEW_YES))
        | reason.str.contains(r"Missing prerequisite: FP", na=False)
    ).sum())
    disagreed = int(results["Eligibility Reason"].str.contains("pathway rules do not agree", na=False).sum()) if not results.empty else 0
    checks = [
        _check("Total evaluated student-course rows", len(evaluated), fail=len(evaluated) != int(remaining["Course Code"].map(lambda value: _text(value) != "").sum())),
        _check("Eligible", eligible),
        _check("Not Eligible", blocked),
        _check("Eligible with Advisor Approval", advisor),
        _check("Courses with no prerequisite", int(type_counts.get(TYPE_NONE, 0))),
        _check("Single-prerequisite rules", int(type_counts.get(TYPE_SINGLE, 0))),
        _check("AND rules", int(type_counts.get(TYPE_AND, 0))),
        _check("OR rules", int(type_counts.get(TYPE_OR, 0))),
        _check("Combined AND/OR rules", int(type_counts.get(TYPE_COMBINED, 0))),
        _check("Co-requisite rules", int(audit["Parse Notes"].astype(str).str.contains("co-requisite", case=False).sum()) if not audit.empty else 0),
        _check("Rules requiring advisor approval", len(exceptions)),
        _check("Rule parse failures", parse_failures, fail=parse_failures > 0 or used_failures > 0),
        _check("Unknown prerequisite course codes", len(unknown), fail=len(unknown) > 0),
        _check("Missing prerequisite-rule rows", missing_rules, fail=missing_rules > 0),
        _check("Spring 2026 leakage", leakage, fail=leakage > 0),
        _check("Duplicate student-course result rows", duplicates, fail=duplicates > 0),
        _check("Conditional-pathway rows clearly marked", len(conditional_output), fail=not conditional_marked),
        _check("Profile students without a remaining-course row", len(absent_profiles)),
        _check("Elective-pool rules missing from Prerequisite Rules", len(output["pool_mismatches"]), review=bool(output["pool_mismatches"])),
        _check("Pathway-unknown students incorrectly evaluated", unknown_evaluated, fail=unknown_evaluated > 0),
        _check("Eligibility statuses outside the official set", unexpected, fail=unexpected > 0),
        _check("Advisor approvals without an official concurrent rule", advisor if not exceptions else 0, fail=advisor > 0 and not exceptions),
        _check("Foundation prerequisite codes referenced", len(foundation)),
        _check("Foundation prerequisite codes found in original transcript", output["foundation_attempts"]["original"]),
        _check("Foundation prerequisite records excluded during preparation", output["foundation_attempts"]["excluded"]),
        _check("Students affected", lacking_students),
        _check("Resolved from historical source", resolved),
        _check("Still lacking evidence", len(lacking), review=len(lacking) > 0),
        _check("Foundation absence incorrectly treated as confirmed failure", incorrect_failure, fail=incorrect_failure > 0),
        _check("Pathway-specific prerequisite outcomes that disagree", disagreed, review=disagreed > 0),
    ]
    population = int(results["Student Code"].nunique()) if not results.empty else 0
    checks.insert(1, _check("Students represented in output", population, fail=population != int(remaining["Student Code"].nunique())))
    return pd.DataFrame(checks)


def export_prerequisite_results(tables: Mapping[str, pd.DataFrame], path: Path) -> None:
    """Write the prerequisite workbook without changing earlier outputs."""

    def write_workbook(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["results"].to_excel(writer, sheet_name="Prerequisite Results", index=False)
            tables["rule_audit"].to_excel(writer, sheet_name="Prerequisite Rule Audit", index=False)
            tables["student_audit"].to_excel(writer, sheet_name="Student Prerequisite Audit", index=False)
            tables["validation"].to_excel(writer, sheet_name="Prerequisite Validation", index=False)
            _format(writer.book["Prerequisite Results"], {"A": 16, "B": 24, "C": 42, "F": 16, "J": 42, "K": 42, "L": 28, "O": 32, "S": 72}, wrap_columns={"J", "K", "Q", "R", "S"})
            _format(writer.book["Prerequisite Rule Audit"], {"A": 16, "E": 42, "F": 42, "G": 42, "K": 55}, wrap_columns={"E", "F", "G", "K"})
            _format(writer.book["Student Prerequisite Audit"], {"A": 16, "B": 16, "C": 24, "D": 55}, wrap_columns={"D"})
            _format(writer.book["Prerequisite Validation"], {"A": 78, "B": 24, "C": 12})

    write_atomic(path, write_workbook)


def _parse_tokens(tokens: list[str]) -> Expression:
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def consume() -> str:
        nonlocal position
        if position >= len(tokens):
            raise PrerequisiteParseError("Prerequisite expression ended early.")
        token = tokens[position]
        position += 1
        return token

    def parse_or() -> Expression:
        node = parse_and()
        items = [node]
        while peek() == "OR":
            consume()
            items.append(parse_and())
        return items[0] if len(items) == 1 else Expression("or", items=tuple(items))

    def parse_and() -> Expression:
        node = parse_factor()
        items = [node]
        while peek() == "AND":
            consume()
            items.append(parse_factor())
        return items[0] if len(items) == 1 else Expression("and", items=tuple(items))

    def parse_factor() -> Expression:
        token = peek()
        if token is None:
            raise PrerequisiteParseError("Prerequisite expression ended early.")
        if token == "(":
            consume()
            node = parse_or()
            if peek() != ")":
                raise PrerequisiteParseError("Prerequisite expression has an unclosed parenthesis.")
            consume()
            return node
        if token in {"AND", "OR", ")", "CO-REQUISITE"}:
            raise PrerequisiteParseError(f"Prerequisite expression has an unexpected {token}.")
        consume()
        return Expression("course", code=token)

    expression = parse_or()
    if peek() is not None:
        raise PrerequisiteParseError(f"Prerequisite expression has trailing text: {peek()}.")
    return expression


def _format_node(expression: Expression, parent: str) -> str:
    if expression.kind == "course":
        return expression.code
    joiner = " AND " if expression.kind == "and" else " OR "
    rendered = joiner.join(_format_node(item, expression.kind) for item in expression.items)
    if parent and parent != expression.kind:
        return f"({rendered})"
    return rendered


def _judge(
    expression: Expression | None,
    passed: set[str],
    unknown: set[str],
    failed: set[str] | None = None,
    attempted: set[str] | None = None,
) -> Judgment:
    failed_codes = failed or set()
    attempted_codes = attempted or set()
    if expression is None:
        return Judgment(True, (), (), (), "No prerequisite required")
    if expression.kind == "course":
        return _judge_course(expression.code, passed, unknown, failed_codes, attempted_codes)
    parts = [_judge(item, passed, unknown, failed_codes, attempted_codes) for item in expression.items]
    unknown_codes = tuple(dict.fromkeys(code for part in parts for code in part.unknown_codes))
    unavailable = tuple(dict.fromkeys(code for part in parts for code in part.unavailable_codes))
    if expression.kind == "and":
        missing = tuple(dict.fromkeys(code for part in parts for code in part.missing_codes))
        satisfied = tuple(dict.fromkeys(code for part in parts for code in part.satisfied_codes))
        if missing or unknown_codes:
            label = ", ".join(missing)
            reason = f"Missing prerequisite: {label}" if len(missing) == 1 else f"Missing prerequisites: {label}"
            if unknown_codes:
                reason = _join_reason(reason, "Unknown prerequisite course code: " + ", ".join(unknown_codes) + ".")
            if unavailable:
                reason = _join_reason(reason, _foundation_reason(unavailable))
            return Judgment(False, satisfied, missing, unknown_codes, reason, unavailable)
        if unavailable:
            return Judgment(False, satisfied, (), unknown_codes, _foundation_reason(unavailable), unavailable)
        return Judgment(True, satisfied, (), (), "Prerequisites " + _natural_join(satisfied) + " passed")
    satisfied_parts = [part for part in parts if part.satisfied]
    if satisfied_parts:
        satisfied = tuple(dict.fromkeys(code for part in satisfied_parts for code in part.satisfied_codes))
        labels = [_branch_label(part) for part in satisfied_parts]
        reason = "Satisfied by " + " and by ".join(labels)
        return Judgment(True, satisfied, (), unknown_codes, reason)
    missing = tuple(dict.fromkeys(code for part in parts for code in part.missing_codes))
    if missing:
        reason = "Missing prerequisites: " + ", ".join(missing)
        if unknown_codes:
            reason = _join_reason(reason, "Unknown prerequisite course code: " + ", ".join(unknown_codes) + ".")
        if unavailable:
            reason = _join_reason(reason, _foundation_reason(unavailable))
        return Judgment(False, (), missing, unknown_codes, reason, unavailable)
    if unavailable:
        return Judgment(False, (), (), unknown_codes, _foundation_reason(unavailable), unavailable)
    reason = "No prerequisite branch is satisfied."
    if unknown_codes:
        reason = _join_reason(reason, "Unknown prerequisite course code: " + ", ".join(unknown_codes) + ".")
    return Judgment(False, (), missing, unknown_codes, reason, unavailable)


def _judge_course(code: str, passed: set[str], unknown: set[str], failed: set[str], attempted: set[str]) -> Judgment:
    if code in unknown and not str(code).startswith("FP"):
        return Judgment(False, (), (), (code,), f"Unknown prerequisite course code: {code}.")
    if code in passed:
        return Judgment(True, (code,), (), (), f"Prerequisite {code} passed")
    if str(code).startswith("FP") and code not in failed and code not in attempted:
        return Judgment(False, (), (), (), _foundation_reason((code,)), (code,))
    return Judgment(False, (), (code,), (), f"Missing prerequisite: {code}")


def _branch_label(part: Judgment) -> str:
    if len(part.satisfied_codes) == 1:
        return part.satisfied_codes[0]
    return "(" + " AND ".join(part.satisfied_codes) + ")"


def _select_rules(rules: list[ParsedRule], record: Mapping[str, object], pools: pd.DataFrame | None = None) -> list[ParsedRule]:
    """Choose the official rules that apply to one remaining course.

    A named elective uses the Prerequisite Rule on its Elective Pools row.
    A study-plan core uses the rule placed on the student's pathway.
    """

    pool_rules = _rules_for_elective_pool(rules, record, pools)
    if pool_rules is not None:
        return _unique_rules(pool_rules)
    scope = _scope_paths(record)
    chosen = [rule for rule in rules if _placement_paths(rule) & scope]
    if not chosen:
        chosen = [rule for rule in rules if not rule.placements]
    if not chosen:
        chosen = list(rules)
    return _unique_rules(chosen)


def _rules_for_elective_pool(
    rules: list[ParsedRule],
    record: Mapping[str, object],
    pools: pd.DataFrame | None,
) -> list[ParsedRule] | None:
    """Return the Elective Pools prerequisite for this course, when the row is an elective."""

    if _text(record.get("Is Elective")) != "Yes" or pools is None or pools.empty:
        return None
    if "Elective Pool" not in pools.columns or "Course Code" not in pools.columns or "Prerequisite Rule" not in pools.columns:
        return None
    names = [part.strip() for part in _text(record.get("Elective Pool")).split("|") if part.strip()]
    course = normalize_course_code(record.get("Course Code"))
    if not names or _missing(course):
        return None
    chosen: list[ParsedRule] = []
    matched_pool = False
    for name in names:
        for pool_row in pools.to_dict(orient="records"):
            if _text(pool_row.get("Elective Pool")) != name:
                continue
            if normalize_course_code(pool_row.get("Course Code")) != course:
                continue
            matched_pool = True
            target = normalize_rule_text(pool_row.get("Prerequisite Rule"))
            chosen.extend(rule for rule in rules if normalize_rule_text(rule.original) == target)
    if not matched_pool or not chosen:
        return None
    return chosen


def _unique_rules(chosen: list[ParsedRule]) -> list[ParsedRule]:
    unique: list[ParsedRule] = []
    seen: set[tuple[object, ...]] = set()
    for rule in sorted(chosen, key=lambda item: (item.original, item.sheet_index)):
        key = (_structure(rule.expression), _structure(rule.co_requisite), rule.parse_status)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def _evaluate_selected(
    course: str,
    rules: list[ParsedRule],
    record: Mapping[str, object],
    history: Mapping[str, object],
    known: set[str],
    exceptions: list[ConcurrentException],
) -> dict[str, object]:
    student = _text(record.get("Student Code"))
    courses = history["students"].get(student, {})
    attempted = set(courses)
    passed = {code for code, attempts in courses.items() if any(item["passed"] for item in attempts)}
    failed = {code for code, attempts in courses.items() if code not in passed and any(item["failed"] for item in attempts)}
    if any(rule.parse_status == "Failed" for rule in rules):
        return {
            "status": "",
            "normalized": " | ".join(rule.normalized for rule in rules),
            "co_requisite": "",
            "rule_type": "",
            "satisfied": [],
            "missing": [],
            "unknown": [],
            "advisor_reason": "",
            "reason": "Prerequisite rule could not be parsed.",
            "referenced": [],
            "evidence_status": "",
            "manual_review": "",
            "unavailable": [],
            "scope": SCOPE_SHARED,
            "original": " | ".join(rule.original for rule in rules),
        }
    evaluations = []
    for rule in rules:
        concurrent = bool(rule.co_requisite_text)
        evaluation = evaluate_prerequisite(
            rule.original,
            passed,
            failed,
            concurrent_corequisite=concurrent,
            concurrent_exceptions=exceptions,
            course_code=course,
            known_codes=known,
            attempted=attempted,
        )
        pathway = _pathway_label(rule, record)
        evaluations.append((pathway, evaluation))
    evaluation = _combine_evaluations(rules, evaluations)
    evaluation["candidate_scope"] = _elective_candidate_scope(record, evaluations)
    return evaluation


def _combine_evaluations(rules: list[ParsedRule], evaluations: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    if len(evaluations) == 1:
        pathway, evaluation = evaluations[0]
        evaluation = dict(evaluation)
        evaluation["scope"] = SCOPE_NONE if evaluation["rule_type"] == TYPE_NONE else SCOPE_SHARED
        evaluation["original"] = rules[0].original
        del pathway
        return evaluation
    statuses = [item[1]["status"] for item in evaluations]
    details = [f"{pathway}: {item['status']}. {item['reason']}" for pathway, item in evaluations]
    originals = [f"{pathway}: {rule.original or 'No prerequisite'}" for (pathway, _), rule in zip(evaluations, rules)]
    normalized = [f"{pathway}: {item['normalized'] or 'No prerequisite'}" for pathway, item in evaluations]
    unknown = sorted({code for _, item in evaluations for code in item["unknown"]})
    referenced = list(dict.fromkeys(code for _, item in evaluations for code in item["referenced"]))
    if all(status == ELIGIBLE for status in statuses):
        status = ELIGIBLE
        reason = "Eligible under every applicable pathway rule. " + " | ".join(details)
    elif all(status == ELIGIBLE_ADVISOR for status in statuses) or (
        set(statuses) <= {ELIGIBLE, ELIGIBLE_ADVISOR} and ELIGIBLE_ADVISOR in statuses
    ):
        status = ELIGIBLE_ADVISOR
        reason = "Advisor approval is required on at least one applicable pathway. " + " | ".join(details)
    elif all(status == NOT_ELIGIBLE for status in statuses):
        status = NOT_ELIGIBLE
        reason = "Not eligible under every applicable pathway rule. " + " | ".join(details)
    else:
        status = NOT_ELIGIBLE
        reason = "The pathway rules do not agree. " + " | ".join(details)
    satisfied = list(dict.fromkeys(code for _, item in evaluations if item["status"] == ELIGIBLE for code in item["satisfied"]))
    missing = []
    for pathway, item in evaluations:
        if item["status"] == NOT_ELIGIBLE:
            for code in item["missing"]:
                label = f"{pathway}: {code}"
                if label not in missing:
                    missing.append(label)
    advisor = " | ".join(item["advisor_reason"] for _, item in evaluations if item["advisor_reason"])
    part_evidence = [item.get("evidence_status", "") for _, item in evaluations]
    if status == NOT_ELIGIBLE and part_evidence and all(value == EVIDENCE_INSUFFICIENT for value in part_evidence):
        evidence_status, manual_review = EVIDENCE_INSUFFICIENT, REVIEW_YES
    elif status:
        evidence_status, manual_review = EVIDENCE_AVAILABLE, REVIEW_NO
    else:
        evidence_status, manual_review = "", ""
    return {
        "status": status,
        "evidence_status": evidence_status,
        "manual_review": manual_review,
        "unavailable": sorted({code for _, item in evaluations for code in item.get("unavailable", [])}),
        "normalized": " | ".join(normalized),
        "co_requisite": " | ".join(item["co_requisite"] for _, item in evaluations if item["co_requisite"]),
        "rule_type": TYPE_COMBINED if len({item["rule_type"] for _, item in evaluations}) > 1 else evaluations[0][1]["rule_type"],
        "satisfied": satisfied,
        "missing": missing,
        "unknown": unknown,
        "advisor_reason": advisor,
        "reason": reason,
        "referenced": referenced,
        "scope": SCOPE_PATHWAY,
        "original": " | ".join(originals),
    }


def _elective_candidate_scope(record: Mapping[str, object], evaluations: list[tuple[str, dict[str, object]]]) -> str:
    """Classify a partial SE/DSAI course without confirming a specialization."""

    if _text(record.get("Pathway Readiness")) != READINESS_PARTIAL:
        return ""
    elective = _text(record.get("Is Elective")) == "Yes"
    se_statuses: list[str] = []
    dsai_statuses: list[str] = []
    for pathway, item in evaluations:
        status = str(item.get("status", ""))
        text = pathway.casefold()
        se = PATH_SE.casefold() in text
        dsai = PATH_DSAI.casefold() in text
        if se and dsai:
            se_statuses.append(status)
            dsai_statuses.append(status)
        elif se:
            se_statuses.append(status)
        elif dsai:
            dsai_statuses.append(status)
    pool = _text(record.get("Elective Pool"))
    both_pools = "SE Major Elective" in pool and "DSAI Major Elective" in pool
    if not se_statuses or not dsai_statuses:
        if elective and both_pools and len(evaluations) == 1 and evaluations[0][1].get("status") == ELIGIBLE:
            return SCOPE_PATHWAY_SAFE
        return ""
    se_ok = all(status == ELIGIBLE for status in se_statuses)
    dsai_ok = all(status == ELIGIBLE for status in dsai_statuses)
    se_blocked = all(status == NOT_ELIGIBLE for status in se_statuses)
    dsai_blocked = all(status == NOT_ELIGIBLE for status in dsai_statuses)
    if se_ok and dsai_ok:
        return SCOPE_PATHWAY_SAFE if elective else SCOPE_PATHWAY_SAFE_CANDIDATE
    if se_ok and dsai_blocked:
        return SCOPE_CONDITIONAL_SE
    if dsai_ok and se_blocked:
        return SCOPE_CONDITIONAL_DSAI
    if se_ok != dsai_ok or set(se_statuses) != set(dsai_statuses):
        return SCOPE_PATHWAY_DEPENDENT
    return ""


def _scope_paths(record: Mapping[str, object]) -> set[str]:
    conditional = _text(record.get("Conditional Pathway"))
    if conditional:
        return {conditional}
    readiness = _text(record.get("Pathway Readiness"))
    if readiness == READINESS_FULL:
        assigned = _text(record.get("Assigned Pathway"))
        blocks = list(PATHWAY_BLOCKS.get(assigned, ())) + list(FUTURE_CONTINUATION.get(assigned, ()))
        return {path for path, _level in blocks}
    if readiness == READINESS_PARTIAL:
        paths: set[str] = set()
        for name in _text(record.get("Possible Pathways")).split("|"):
            blocks = PATHWAY_BLOCKS.get(name.strip(), ())
            paths.update(path for path, _level in blocks)
        return paths or {PATH_SE, PATH_DSAI}
    return set()


def _placements_for(plan: pd.DataFrame, course: str, original: str) -> tuple[tuple[str, str, str], ...]:
    if course == "":
        return ()
    target = normalize_rule_text(original)
    found: list[tuple[str, str, str]] = []
    for record in plan.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code) or str(code) != course:
            continue
        if normalize_rule_text(record.get("Prerequisite Rule")) != target:
            continue
        found.append((_text(record.get("Specialization Path")), _text(record.get("Level")), _text(record.get("Year"))))
    return tuple(found)


def _rule_audit_row(rule: ParsedRule) -> dict[str, object]:
    paths = sorted({path for path, _level, _year in rule.placements})
    levels = sorted({level for _path, level, _year in rule.placements})
    return {
        "Course Code": rule.course_code,
        "Course Title": rule.course_title,
        "Specialization Path": " | ".join(paths),
        "Level": " | ".join(levels),
        "Original Rule": rule.original,
        "Normalized Rule": rule.normalized,
        "Parsed Expression": format_expression(rule.expression),
        "Referenced Course Codes": ", ".join(rule.referenced),
        "Rule Type": rule.rule_type,
        "Parse Status": rule.parse_status,
        "Parse Notes": rule.parse_notes,
    }


def _result_row(record: Mapping[str, object], rules: list[ParsedRule], evaluation: dict[str, object]) -> dict[str, object]:
    row = _base_row(record)
    row.update({
        "Prerequisite Rule Original": evaluation.get("original", rules[0].original),
        "Prerequisite Rule Normalized": evaluation["normalized"],
        "Co-requisite Rule": evaluation["co_requisite"],
        "Rule Type": evaluation["rule_type"],
        "Prerequisite Scope": evaluation["scope"],
        "Eligibility Status": evaluation["status"],
        "Eligibility Evidence Status": evaluation.get("evidence_status", ""),
        "Manual Review Required": evaluation.get("manual_review", ""),
        "Satisfied Prerequisites": "; ".join(evaluation["satisfied"]),
        "Missing Prerequisites": "; ".join(evaluation["missing"]),
        "Advisor Approval Reason": evaluation["advisor_reason"],
        "Eligibility Reason": evaluation["reason"],
        "Candidate Scope": evaluation.get("candidate_scope", ""),
    })
    return row


def _audit_rows(record: Mapping[str, object], evaluation: dict[str, object], history: Mapping[str, object]) -> list[dict[str, object]]:
    student = _text(record.get("Student Code"))
    course = _text(record.get("Course Code"))
    courses = history["students"].get(student, {})
    rows = []
    for code in evaluation["referenced"]:
        attempts = courses.get(code, [])
        passed = any(item["passed"] for item in attempts)
        unavailable = set(evaluation.get("unavailable", []))
        if code in evaluation["unknown"]:
            result = "Unknown code"
            attempt_text = "; ".join(item["label"] for item in attempts) if attempts else "Not taken"
        elif passed:
            result = "Passed"
            attempt_text = "; ".join(item["label"] for item in attempts)
        elif code in unavailable:
            result = "Evidence unavailable"
            attempt_text = "Not recorded in the pre-Spring 2026 transcript"
        elif attempts:
            result = "Not passed"
            attempt_text = "; ".join(item["label"] for item in attempts)
        else:
            result = "Not taken"
            attempt_text = "Not taken"
        rows.append({
            "Student Code": student,
            "Course Code": course,
            "Referenced Prerequisite": code,
            "Historical Attempts": attempt_text,
            "Passed Before Holdout": "Yes" if passed else "No",
            "Evaluation Result": result,
        })
    return rows


def _pathway_row(record: Mapping[str, object]) -> dict[str, object]:
    row = _base_row(record)
    row.update({
        "Course Code": "",
        "Eligibility Status": PATHWAY_REQUIRED,
        "Eligibility Evidence Status": "",
        "Manual Review Required": "",
        "Eligibility Reason": PATHWAY_REQUIRED,
    })
    return row


def _missing_rule_row(record: Mapping[str, object]) -> dict[str, object]:
    row = _base_row(record)
    row.update({"Eligibility Reason": "Missing source rule", "Prerequisite Scope": "Missing source rule"})
    return row


def _base_row(record: Mapping[str, object]) -> dict[str, object]:
    return {column: record.get(column, "") for column in RESULT_COLUMNS}


def _unmatched_pool_rules(pools: pd.DataFrame, by_code: dict[str, list[ParsedRule]]) -> list[str]:
    """Return elective-pool rules that do not match any Prerequisite Rules row."""

    if pools.empty or "Course Code" not in pools.columns or "Prerequisite Rule" not in pools.columns:
        return []
    missing: list[str] = []
    for record in pools.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code):
            continue
        target = normalize_rule_text(record.get("Prerequisite Rule"))
        sheet_rules = by_code.get(str(code), [])
        if any(normalize_rule_text(rule.original) == target for rule in sheet_rules):
            continue
        pool = _text(record.get("Elective Pool")) or "Elective Pools"
        missing.append(f"{pool}: {code} {target or 'has no matching prerequisite row'}")
    return missing


def _known_codes(
    plan: pd.DataFrame,
    pools: pd.DataFrame,
    rules: pd.DataFrame,
    transcript: pd.DataFrame,
    advising: pd.DataFrame,
    validation_lists: pd.DataFrame,
) -> set[str]:
    codes: set[str] = set()
    for frame, column in ((plan, "Course Code"), (pools, "Course Code"), (rules, "Course Code"), (transcript, "Course Code")):
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            code = normalize_course_code(value)
            if not _missing(code) and COURSE_PATTERN.fullmatch(str(code)):
                codes.add(str(code))
    if "Allowed Value" in validation_lists.columns:
        for value in validation_lists["Allowed Value"].tolist():
            code = normalize_course_code(value)
            if not _missing(code) and COURSE_PATTERN.fullmatch(str(code)):
                codes.add(str(code))
    advising_text = " ".join(advising.get("Rule", pd.Series(dtype=str)).astype(str))
    if "FP" in advising_text.upper():
        referenced = COURSE_PATTERN.findall(" ".join(rules.get("Prerequisite Rule", pd.Series(dtype=str)).dropna().astype(str).str.upper()))
        codes.update(code for code in referenced if code.startswith("FP"))
    return codes


def _unknown_codes(expression: Expression | None, other: Expression | None, known: set[str] | None) -> set[str]:
    if known is None:
        return set()
    return {code for code in [*referenced_codes(expression), *referenced_codes(other)] if code not in known}


def _exception_codes(course: str, exceptions: list[ConcurrentException], failed: set[str]) -> set[str]:
    return {item.prerequisite_code for item in exceptions if item.course_code == course and item.prerequisite_code in failed}


def _placement_paths(rule: ParsedRule) -> set[str]:
    return {path for path, _level, _year in rule.placements}


def _pathway_label(rule: ParsedRule, record: Mapping[str, object]) -> str:
    conditional = _text(record.get("Conditional Pathway"))
    if conditional:
        return conditional
    paths = sorted(_placement_paths(rule))
    return " | ".join(paths) if paths else _text(record.get("Assigned Pathway"))


def _structure(expression: Expression | None) -> tuple[object, ...]:
    if expression is None:
        return ("none",)
    if expression.kind == "course":
        return ("course", expression.code)
    children = tuple(sorted(_structure(item) for item in expression.items))
    return (expression.kind, children)


def _used_parse_failures(evaluated: pd.DataFrame, audit: pd.DataFrame) -> int:
    failed_codes = set(audit.loc[audit["Parse Status"].eq("Failed"), "Course Code"].astype(str))
    if not failed_codes or evaluated.empty:
        return 0
    return int(evaluated["Course Code"].astype(str).isin(failed_codes).sum())


def _evidence_fields(status: str, missing: list[str], unavailable: list[str]) -> tuple[str, str]:
    if status in {ELIGIBLE, ELIGIBLE_ADVISOR}:
        return EVIDENCE_AVAILABLE, REVIEW_NO
    if status == NOT_ELIGIBLE and unavailable and not missing:
        return EVIDENCE_INSUFFICIENT, REVIEW_YES
    if status == NOT_ELIGIBLE:
        return EVIDENCE_AVAILABLE, REVIEW_NO
    return "", ""


def _foundation_reason(codes: tuple[str, ...] | list[str]) -> str:
    del codes
    return (
        "The prerequisite is required by the project reference Excel, "
        "but the available transcript data does not contain student-level evidence for that foundation course."
    )


def _copy_foundation_attempts(history: dict[str, object], frame: pd.DataFrame | None) -> None:
    """Add pre-holdout FP attempts to the prerequisite index only."""

    if frame is None or frame.empty or "Course Code" not in frame.columns:
        return
    required = ["Student Code", "Course Code", "Semester", "Academic Year", "Attempt Number", "Is Passed", "Is Failed", "Is Withdrawn"]
    if any(column not in frame.columns for column in required):
        return
    holdout = frame["Semester"].astype(str).str.contains(HOLDOUT_TOKEN) | frame["Academic Year"].astype(str).str.contains(HOLDOUT_TOKEN)
    students = history["students"]
    for record in frame.loc[~holdout].to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code) or not str(code).startswith("FP"):
            continue
        student = _text(record.get("Student Code"))
        if student == "":
            continue
        course_code = str(code)
        attempt = {
            "year": _year_value(record.get("Academic Year")),
            "term": _term_value(record.get("Term", record.get("Semester"))),
            "attempt": _attempt_value(record.get("Attempt Number")),
            "passed": _flag(record.get("Is Passed")),
            "failed": _flag(record.get("Is Failed")),
            "withdrawn": _flag(record.get("Is Withdrawn")),
            "label": _attempt_label(record),
        }
        bucket = students.setdefault(student, {}).setdefault(course_code, [])
        bucket.append(attempt)
    for courses in students.values():
        for attempts in courses.values():
            attempts.sort(key=lambda item: (item["year"], item["term"], item["attempt"]))


def _foundation_attempt_count(frame: pd.DataFrame | None, codes: list[str]) -> int:
    if frame is None or frame.empty or "Course Code" not in frame.columns or not codes:
        return 0
    normalized = frame["Course Code"].map(lambda value: str(normalize_course_code(value) or ""))
    return int(normalized.isin(codes).sum())


def _resolved_foundation_rows(audit: pd.DataFrame, codes: list[str]) -> int:
    if audit.empty or not codes:
        return 0
    matched = audit.loc[audit["Referenced Prerequisite"].isin(codes) & audit["Evaluation Result"].eq("Passed")]
    if matched.empty:
        return 0
    return int(matched.groupby(["Student Code", "Course Code"]).ngroups)


def _split_corequisite(text: str) -> tuple[str, str]:
    parts = re.split(r"\bCO-REQUISITE\b", text, maxsplit=1)
    if len(parts) == 1:
        return text.strip(), ""
    return parts[0].strip(" ;,|"), parts[1].strip(" ;,|:")


def _is_concurrent_approval_text(text: str) -> bool:
    markers = ("TOGETHER", "CONCURRENT", "CO-REQUISITE")
    return "ADVISOR" in text and any(marker in text for marker in markers)


def _collect_codes(expression: Expression, found: list[str]) -> None:
    if expression.kind == "course":
        found.append(expression.code)
        return
    for item in expression.items:
        _collect_codes(item, found)


def _collect_operators(expression: Expression, found: set[str]) -> None:
    if expression.kind == "course":
        return
    found.add(expression.kind)
    for item in expression.items:
        _collect_operators(item, found)


def _attempt_label(record: Mapping[str, object]) -> str:
    if _flag(record.get("Is Passed")):
        label = "Passed"
    elif _flag(record.get("Is Failed")):
        label = "Failed"
    elif _flag(record.get("Is Withdrawn")):
        label = "Withdrawn"
    else:
        label = "Other"
    if _flag(record.get("Is Repeated For Improvement")) or _text(record.get("Remarks")).upper() == "N":
        label += "; repeated to improve GPA"
    semester = _text(record.get("Semester")) or "semester not recorded"
    return f"{semester} attempt {_attempt_value(record.get('Attempt Number'))}: {label}"


def _join_rule(prerequisite: str, co_requisite: str) -> str:
    if prerequisite and co_requisite:
        return f"{prerequisite} | CO-REQUISITE {co_requisite}"
    return prerequisite or (f"CO-REQUISITE {co_requisite}" if co_requisite else "")


def _join_reason(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _natural_join(values: tuple[str, ...] | list[str]) -> str:
    items = list(values)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


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


def _missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return value is None


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
