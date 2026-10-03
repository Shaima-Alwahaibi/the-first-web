"""Calculate official course difficulty and a leakage-safe historical feature table.

Pass, fail, withdrawal, and repeat flags come from the prepared transcript.
The official Easy, Medium, and Hard label is determined only by pass rate.
Spring 2026 remains in the descriptive table and stays out of the modeling table.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from preparation_checks import (
    code_ojt_status,
    is_foundation_code,
    normalize_course_code,
    normalize_course_title,
    title_ojt_status,
    write_atomic,
)
from repeat_classification import (
    OUTCOME_FAILED_RECOVERED,
    OUTCOME_MANUAL,
    OUTCOME_POSTPONED,
    REPEAT_FAILURE,
    REPEAT_IMPROVEMENT,
    REPEAT_UNRESOLVED,
    REPEAT_WITHDRAWAL,
    derive_repeat_classification,
)
from study_plan_matching import HOLDOUT_TOKEN

SCOPE_ALL = "All Transcript Records"
PURPOSE_ALL = "Descriptive Task 9 / Paper Analysis"
SCOPE_HISTORICAL = "Historical Through Fall 2025"
PURPOSE_HISTORICAL = "Modeling Feature Table"
FEATURE_CUTOFF = "Fall 2025"
LEVEL_EASY = "Easy"
LEVEL_MEDIUM = "Medium"
LEVEL_HARD = "Hard"
WARNING_YES = "Yes"
WARNING_NO = "No"
# Analytical reliability flag only. It does not change Easy, Medium, or Hard.
CATEGORY_PLAN = "Study Plan Course"
CATEGORY_POOL = "Elective Pool Course"
CATEGORY_GENERAL = "General Requirement / Reference Gap"
CATEGORY_HISTORICAL = "Historical / Off-Plan Course"
CATEGORY_UNKNOWN = "Truly Unknown Course"
GENERAL_REQUIREMENT_GAP = {"UNWE2216", "UNOC2207", "UNOS2207", "UNIS2206", "UNFL2213"}
LOW_SAMPLE_ATTEMPTS = 30
MODERATE_SAMPLE_ATTEMPTS = 100
GRADE_POINT_MINIMUM = 0.0
GRADE_POINT_MAXIMUM = 4.0

DIFFICULTY_COLUMNS = [
    "Course Code",
    "Course Title",
    "Total Attempts",
    "Unique Students",
    "Pass Count",
    "Fail Count",
    "Withdrawn Count",
    "Other / Unclassified Attempt Count",
    "Pass Rate",
    "Fail Rate",
    "Withdrawal Rate",
    "Average Grade Point",
    "Median Grade Point",
    "Grade Point Standard Deviation",
    "Grade Point Observation Count",
    "Repeat Count",
    "Repeat Due to Failure Count",
    "Repeat After Withdrawal Count",
    "Repeat for Improvement Count",
    "Repeat Type Unresolved Count",
    "Difficulty Level",
    "Analysis Scope",
    "Low Sample Warning",
    "Difficulty Evidence Quality",
    "Specialization Paths",
    "Level",
    "Year",
    "Course Type",
    "Requirement Type",
    "Credit Hours",
    "Title Aliases",
    "Analysis Purpose",
    "Course Reference Category",
]
HISTORICAL_COLUMNS = [
    "Course Code",
    "Course Title",
    "Historical Total Attempts",
    "Historical Unique Students",
    "Historical Pass Count",
    "Historical Fail Count",
    "Historical Withdrawn Count",
    "Historical Other / Unclassified Attempt Count",
    "Historical Pass Rate",
    "Historical Fail Rate",
    "Historical Withdrawal Rate",
    "Historical Average Grade Point",
    "Historical Median Grade Point",
    "Historical Grade Point Std",
    "Historical Grade Point Observation Count",
    "Historical Repeat Count",
    "Historical Repeat Due to Failure Count",
    "Historical Repeat After Withdrawal Count",
    "Historical Repeat for Improvement Count",
    "Historical Repeat Type Unresolved Count",
    "Historical Difficulty Level",
    "Historical Low Sample Warning",
    "Feature Cutoff",
]
SCHEMA_FIELDS = [
    "Student Code",
    "Course Code",
    "Course Name",
    "Course Title",
    "Academic Year",
    "Term",
    "Semester",
    "Attempt Number",
    "Grade",
    "Grade Point",
    "Result",
    "Credit Hours",
    "Is Passed",
    "Is Failed",
    "Is Withdrawn",
    "Is Repeated",
    "Is Repeated Due To Failure",
    "Is Repeated For Improvement",
    "Remarks",
]


def describe_difficulty_schema(frame: pd.DataFrame) -> dict[str, object]:
    """Report which Task 9 fields are present on a source table."""

    columns = list(frame.columns)
    return {
        "rows": int(len(frame)),
        "columns": columns,
        "present": [field for field in SCHEMA_FIELDS if field in frame.columns],
        "missing": [field for field in SCHEMA_FIELDS if field not in frame.columns],
    }


def difficulty_level(pass_count: int, attempts: int) -> str:
    """Return Easy, Medium, or Hard from the official pass-rate thresholds.

    The comparison uses integer cross-multiplication so an exact 85 percent
    or 65 percent boundary is not changed by binary floating point.
    """

    if attempts <= 0:
        raise ValueError("Difficulty level requires at least one attempt.")
    if pass_count * 100 >= 85 * attempts:
        return LEVEL_EASY
    if pass_count * 100 >= 65 * attempts:
        return LEVEL_MEDIUM
    return LEVEL_HARD


def build_difficulty_output(
    transcript: pd.DataFrame,
    study_plan: pd.DataFrame | None = None,
    elective_pools: pd.DataFrame | None = None,
    *,
    historical_reference_rows: int | None = None,
    holdout_reference_rows: int | None = None,
) -> dict[str, object]:
    """Build the descriptive table, historical features, audits, and validation."""

    attempts = _prepare_attempts(transcript)
    valid = attempts.loc[~attempts["_ojt"]].copy()
    descriptive = valid.copy()
    historical = valid.loc[~valid["_spring_2026"] & ~valid["_foundation"]].copy()
    spring = valid.loc[valid["_spring_2026"] & ~valid["_foundation"]].copy()
    catalog = _plan_catalog(study_plan)
    pool_codes = _pool_codes(_official_pools(elective_pools))
    difficulty = _course_table(descriptive, catalog, pool_codes, scope=SCOPE_ALL, purpose=PURPOSE_ALL)
    features = _historical_table(historical, catalog, pool_codes)
    holdout = _holdout_audit(historical, spring, descriptive, catalog)
    repeats = _repeat_audit(difficulty)
    scope = _scope_audit(
        attempts,
        valid,
        historical,
        spring,
        difficulty,
        historical_reference_rows=historical_reference_rows,
        holdout_reference_rows=holdout_reference_rows,
    )
    summary = _paper_summary(difficulty, valid)
    outcomes = _outcome_audit(valid)
    mismatches = _mismatch_audit(valid)
    validation = _validate(
        attempts,
        valid,
        difficulty,
        features,
        historical,
        catalog,
        pool_codes,
        historical_reference_rows=historical_reference_rows,
        holdout_reference_rows=holdout_reference_rows,
    )
    return {
        "attempts": attempts,
        "difficulty": difficulty,
        "features": features,
        "holdout": holdout,
        "repeats": repeats,
        "scope": scope,
        "summary": summary,
        "outcomes": outcomes,
        "mismatches": mismatches,
        "validation": validation,
        "spring_2026_rows_in_historical_features": int(historical["_spring_2026"].sum()) if not historical.empty else 0,
    }


def export_difficulty_results(tables: dict[str, object], path: Path) -> None:
    """Write the course-difficulty workbook without changing earlier outputs."""

    def write_workbook(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["difficulty"].to_excel(writer, sheet_name="Course Difficulty", index=False)
            tables["features"].to_excel(writer, sheet_name="Historical Difficulty Features", index=False)
            tables["holdout"].to_excel(writer, sheet_name="Holdout Difficulty Audit", index=False)
            tables["repeats"].to_excel(writer, sheet_name="Repeat Difficulty Audit", index=False)
            tables["scope"].to_excel(writer, sheet_name="Difficulty Scope Audit", index=False)
            tables["validation"].to_excel(writer, sheet_name="Difficulty Validation", index=False)
            tables["summary"].to_excel(writer, sheet_name="Paper Summary", index=False)
            tables["outcomes"].to_excel(writer, sheet_name="Attempt Outcome Audit", index=False)
            tables["mismatches"].to_excel(writer, sheet_name="Repeat Flag Mismatch", index=False)
            _format(writer.book["Course Difficulty"], {"A": 16, "B": 42, "U": 28, "AC": 42, "AD": 36}, wrap_columns={"B", "AC", "AD"})
            _format(writer.book["Historical Difficulty Features"], {"A": 16, "B": 42}, wrap_columns={"B"})
            _format(writer.book["Holdout Difficulty Audit"], {"A": 16, "B": 42}, wrap_columns={"B"})
            _format(writer.book["Repeat Difficulty Audit"], {"A": 16, "B": 42, "H": 42}, wrap_columns={"B", "H"})
            _format(writer.book["Difficulty Scope Audit"], {"A": 62, "B": 88}, wrap_columns={"B"})
            _format(writer.book["Difficulty Validation"], {"A": 78, "B": 18, "C": 12})
            _format(writer.book["Paper Summary"], {"A": 42, "B": 72}, wrap_columns={"B"})

    write_atomic(path, write_workbook)


def save_difficulty_figures(difficulty: pd.DataFrame, directory: Path) -> list[Path]:
    """Save the paper figures and the plotting tables behind them."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    directory = Path(directory)
    data_directory = directory / "data"
    directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    frame = difficulty.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=DIFFICULTY_COLUMNS)
    saved: list[Path] = []
    saved.extend(_figure_pass_rate_histogram(frame, directory, data_directory, plt))
    saved.extend(_figure_level_counts(frame, directory, data_directory, plt))
    saved.extend(_figure_course_bars(frame, directory, data_directory, plt))
    saved.extend(_figure_grade_scatter(frame, directory, data_directory, plt))
    saved.extend(_figure_repeat_scatter(frame, directory, data_directory, plt))
    saved.extend(_figure_withdrawal_bars(frame, directory, data_directory, plt))
    return saved


def _prepare_attempts(transcript: pd.DataFrame) -> pd.DataFrame:
    _require_columns(transcript, ["Course Code", "Is Passed", "Is Failed", "Is Withdrawn"], "Transcript attempts")
    frame = transcript.copy()
    frame["_code"] = frame["Course Code"].map(_code)
    name_column = "Course Name" if "Course Name" in frame.columns else "Course Title" if "Course Title" in frame.columns else ""
    frame["_name"] = frame[name_column].map(_text) if name_column else ""
    frame["_passed"] = frame["Is Passed"].map(_flag)
    frame["_failed"] = frame["Is Failed"].map(_flag)
    frame["_withdrawn"] = frame["Is Withdrawn"].map(_flag)
    frame["_failure_repeat"] = frame["Is Repeated Due To Failure"].map(_flag) if "Is Repeated Due To Failure" in frame.columns else False
    frame["_improvement_repeat"] = frame["Is Repeated For Improvement"].map(_flag) if "Is Repeated For Improvement" in frame.columns else False
    frame["_attempt_number"] = frame["Attempt Number"].map(_attempt_value) if "Attempt Number" in frame.columns else 1
    frame["_grade_point"] = frame["Grade Point"].map(_grade_point) if "Grade Point" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    frame["_grade_point_valid"] = frame["_grade_point"].map(lambda value: value is not None and GRADE_POINT_MINIMUM <= value <= GRADE_POINT_MAXIMUM)
    frame["_grade_point_invalid"] = frame["Grade Point"].map(_invalid_grade_point) if "Grade Point" in frame.columns else pd.Series([False] * len(frame), index=frame.index)
    frame["_student"] = frame["Student Code"].map(_text) if "Student Code" in frame.columns else ""
    frame["_spring_2026"] = _spring_mask(frame)
    frame["_foundation"] = frame["_code"].map(lambda code: bool(code) and is_foundation_code(code))
    frame["_ojt"] = [
        _is_ojt(code, name)
        for code, name in zip(frame["_code"], frame["_name"], strict=True)
    ]
    classified = derive_repeat_classification(frame.loc[~frame["_ojt"]].drop(columns=[column for column in frame.columns if column.startswith("_")], errors="ignore"))
    frame["_derived_repeat"] = ""
    frame["_derived_outcome"] = ""
    frame["_repeat_mismatch"] = ""
    if not classified.empty:
        frame.loc[classified.index, "_derived_repeat"] = classified["Derived Repeat Type"].to_numpy()
        frame.loc[classified.index, "_derived_outcome"] = classified["Derived Academic Outcome"].to_numpy()
        frame.loc[classified.index, "_repeat_mismatch"] = classified["Repeat Flag Mismatch"].to_numpy()
    frame["_failure_repeat"] = frame["_derived_repeat"].eq(REPEAT_FAILURE)
    frame["_withdrawal_repeat"] = frame["_derived_repeat"].eq(REPEAT_WITHDRAWAL)
    frame["_improvement_repeat"] = frame["_derived_repeat"].eq(REPEAT_IMPROVEMENT)
    frame["_unresolved_repeat"] = frame["_derived_repeat"].eq(REPEAT_UNRESOLVED)
    frame["_repeat"] = frame["_derived_repeat"].isin([REPEAT_FAILURE, REPEAT_WITHDRAWAL, REPEAT_IMPROVEMENT, REPEAT_UNRESOLVED])
    frame["_other"] = ~frame["_passed"] & ~frame["_failed"] & ~frame["_withdrawn"]
    frame["_credit"] = frame["Credit Hours"].map(_number) if "Credit Hours" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    return frame


def _course_table(frame: pd.DataFrame, catalog: dict[str, dict[str, object]], pool_codes: set[str], *, scope: str, purpose: str) -> pd.DataFrame:
    rows = [_course_row(code, group, catalog, pool_codes, scope=scope, purpose=purpose) for code, group in _groups(frame)]
    return _frame(rows, DIFFICULTY_COLUMNS, ["Course Code"])


def _historical_table(frame: pd.DataFrame, catalog: dict[str, dict[str, object]], pool_codes: set[str]) -> pd.DataFrame:
    del pool_codes
    rows = []
    for code, group in _groups(frame):
        metrics = _metrics(group)
        identity = _identity(code, group, catalog)
        rows.append({
            "Course Code": code,
            "Course Title": identity["title"],
            "Historical Total Attempts": metrics["attempts"],
            "Historical Unique Students": metrics["students"],
            "Historical Pass Count": metrics["passed"],
            "Historical Fail Count": metrics["failed"],
            "Historical Withdrawn Count": metrics["withdrawn"],
            "Historical Other / Unclassified Attempt Count": metrics["other"],
            "Historical Pass Rate": metrics["pass_rate"],
            "Historical Fail Rate": metrics["fail_rate"],
            "Historical Withdrawal Rate": metrics["withdrawal_rate"],
            "Historical Average Grade Point": metrics["average"],
            "Historical Median Grade Point": metrics["median"],
            "Historical Grade Point Std": metrics["std"],
            "Historical Grade Point Observation Count": metrics["observations"],
            "Historical Repeat Count": metrics["repeats"],
            "Historical Repeat Due to Failure Count": metrics["failure_repeats"],
            "Historical Repeat After Withdrawal Count": metrics["withdrawal_repeats"],
            "Historical Repeat for Improvement Count": metrics["improvement_repeats"],
            "Historical Repeat Type Unresolved Count": metrics["unresolved_repeats"],
            "Historical Difficulty Level": difficulty_level(metrics["passed"], metrics["attempts"]),
            "Historical Low Sample Warning": WARNING_YES if metrics["attempts"] < LOW_SAMPLE_ATTEMPTS else WARNING_NO,
            "Feature Cutoff": FEATURE_CUTOFF,
        })
    return _frame(rows, HISTORICAL_COLUMNS, ["Course Code"])


def _course_row(code: str, group: pd.DataFrame, catalog: dict[str, dict[str, object]], pool_codes: set[str], *, scope: str, purpose: str) -> dict[str, object]:
    metrics = _metrics(group)
    identity = _identity(code, group, catalog)
    level = difficulty_level(metrics["passed"], metrics["attempts"])
    return {
        "Course Code": code,
        "Course Title": identity["title"],
        "Total Attempts": metrics["attempts"],
        "Unique Students": metrics["students"],
        "Pass Count": metrics["passed"],
        "Fail Count": metrics["failed"],
        "Withdrawn Count": metrics["withdrawn"],
        "Other / Unclassified Attempt Count": metrics["other"],
        "Pass Rate": metrics["pass_rate"],
        "Fail Rate": metrics["fail_rate"],
        "Withdrawal Rate": metrics["withdrawal_rate"],
        "Average Grade Point": metrics["average"],
        "Median Grade Point": metrics["median"],
        "Grade Point Standard Deviation": metrics["std"],
        "Grade Point Observation Count": metrics["observations"],
        "Repeat Count": metrics["repeats"],
        "Repeat Due to Failure Count": metrics["failure_repeats"],
        "Repeat After Withdrawal Count": metrics["withdrawal_repeats"],
        "Repeat for Improvement Count": metrics["improvement_repeats"],
        "Repeat Type Unresolved Count": metrics["unresolved_repeats"],
        "Difficulty Level": level,
        "Analysis Scope": scope,
        "Low Sample Warning": WARNING_YES if metrics["attempts"] < LOW_SAMPLE_ATTEMPTS else WARNING_NO,
        "Difficulty Evidence Quality": _evidence_quality(metrics["attempts"]),
        "Specialization Paths": identity["paths"],
        "Level": identity["level"],
        "Year": identity["year"],
        "Course Type": identity["course_type"],
        "Requirement Type": identity["requirement_type"],
        "Credit Hours": identity["credits"] if identity["credits"] is not None else _single_number(group["_credit"]),
        "Title Aliases": identity["aliases"],
        "Analysis Purpose": purpose,
        "Course Reference Category": _reference_category(code, code in catalog, code in pool_codes),
    }


def _metrics(group: pd.DataFrame) -> dict[str, object]:
    attempts = int(len(group))
    passed = int(group["_passed"].sum())
    failed = int(group["_failed"].sum())
    withdrawn = int(group["_withdrawn"].sum())
    other = int(group["_other"].sum())
    points = [value for value, valid in zip(group["_grade_point"], group["_grade_point_valid"], strict=True) if valid]
    return {
        "attempts": attempts,
        "students": int(group["_student"].replace("", pd.NA).nunique(dropna=True)),
        "passed": passed,
        "failed": failed,
        "withdrawn": withdrawn,
        "other": other,
        "pass_rate": passed / attempts,
        "fail_rate": failed / attempts,
        "withdrawal_rate": withdrawn / attempts,
        "average": sum(points) / len(points) if points else None,
        "median": _median(points),
        "std": _std(points),
        "observations": len(points),
        "repeats": int(group["_repeat"].sum()),
        "failure_repeats": int(group["_failure_repeat"].sum()),
        "withdrawal_repeats": int(group["_withdrawal_repeat"].sum()),
        "improvement_repeats": int(group["_improvement_repeat"].sum()),
        "unresolved_repeats": int(group["_unresolved_repeat"].sum()),
    }


def _identity(code: str, group: pd.DataFrame, catalog: dict[str, dict[str, object]]) -> dict[str, object]:
    meta = catalog.get(code, {})
    plan_title = _text(meta.get("title"))
    observed = sorted({name for name in group["_name"].map(_text) if name})
    title = plan_title or (observed[0] if len(observed) == 1 else " | ".join(observed))
    aliases = [name for name in observed if name and name.casefold() != title.casefold()]
    return {
        "title": title,
        "aliases": " | ".join(aliases),
        "paths": _text(meta.get("paths")),
        "level": _text(meta.get("level")),
        "year": _text(meta.get("year")),
        "course_type": _text(meta.get("course_type")),
        "requirement_type": _text(meta.get("requirement_type")),
        "credits": meta.get("credits"),
    }


def _holdout_audit(
    historical: pd.DataFrame,
    spring: pd.DataFrame,
    descriptive: pd.DataFrame,
    catalog: dict[str, dict[str, object]],
) -> pd.DataFrame:
    historical_metrics = {code: _metrics(group) for code, group in _groups(historical)}
    spring_metrics = {code: _metrics(group) for code, group in _groups(spring)}
    all_metrics = {code: _metrics(group) for code, group in _groups(descriptive)}
    rows = []
    for code in sorted(set(historical_metrics) | set(spring_metrics) | set(all_metrics)):
        historical_row = historical_metrics.get(code)
        spring_row = spring_metrics.get(code)
        all_row = all_metrics.get(code)
        historical_level = difficulty_level(historical_row["passed"], historical_row["attempts"]) if historical_row else ""
        all_level = difficulty_level(all_row["passed"], all_row["attempts"]) if all_row else ""
        changed = ""
        if historical_level and all_level:
            changed = WARNING_YES if historical_level != all_level else WARNING_NO
        title = ""
        if code in catalog and _text(catalog[code].get("title")):
            title = _text(catalog[code].get("title"))
        elif all_row is not None:
            title = _identity(code, descriptive.loc[descriptive["_code"].eq(code)], catalog)["title"]
        rows.append({
            "Course Code": code,
            "Course Title": title,
            "Historical Attempts": historical_row["attempts"] if historical_row else 0,
            "Historical Pass Rate": historical_row["pass_rate"] if historical_row else None,
            "Spring 2026 Attempts": spring_row["attempts"] if spring_row else 0,
            "Spring 2026 Pass Rate": spring_row["pass_rate"] if spring_row else None,
            "All-Record Pass Rate": all_row["pass_rate"] if all_row else None,
            "Historical Difficulty Level": historical_level,
            "All-Record Difficulty Level": all_level,
            "Difficulty Category Changed": changed,
        })
    columns = [
        "Course Code", "Course Title", "Historical Attempts", "Historical Pass Rate",
        "Spring 2026 Attempts", "Spring 2026 Pass Rate", "All-Record Pass Rate",
        "Historical Difficulty Level", "All-Record Difficulty Level", "Difficulty Category Changed",
    ]
    return _frame(rows, columns, ["Course Code"])


def _repeat_audit(difficulty: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in difficulty.iterrows():
        attempts = int(row["Total Attempts"])
        repeats = int(row["Repeat Count"])
        rows.append({
            "Course Code": row["Course Code"],
            "Course Title": row["Course Title"],
            "Total Repeat Attempts": repeats,
            "Repeat Due to Failure": int(row["Repeat Due to Failure Count"]),
            "Repeat After Withdrawal": int(row["Repeat After Withdrawal Count"]),
            "Repeat for Improvement": int(row["Repeat for Improvement Count"]),
            "Unresolved Repeat": int(row["Repeat Type Unresolved Count"]),
            "Repeat Rate": repeats / attempts if attempts else None,
            "Denominator": "Repeat Count / Total Attempts. Repeat Count is the chronological derived type: failure, withdrawal, improvement, or unresolved. Stored repeat flags stay on the source row for audit.",
        })
    columns = [
        "Course Code", "Course Title", "Total Repeat Attempts", "Repeat Due to Failure",
        "Repeat After Withdrawal", "Repeat for Improvement", "Unresolved Repeat", "Repeat Rate", "Denominator",
    ]
    return _frame(rows, columns, ["Course Code"])


def _scope_audit(
    attempts: pd.DataFrame,
    valid: pd.DataFrame,
    historical: pd.DataFrame,
    spring: pd.DataFrame,
    difficulty: pd.DataFrame,
    *,
    historical_reference_rows: int | None,
    holdout_reference_rows: int | None,
) -> pd.DataFrame:
    raw = int(len(attempts))
    ojt = int(attempts["_ojt"].sum())
    foundation = int((~attempts["_ojt"] & attempts["_foundation"]).sum())
    spring_rows = int(attempts["_spring_2026"].sum())
    historical_rows = raw - spring_rows
    missing_code = int(attempts["_code"].eq("").sum())
    unclassified = int((~attempts["_ojt"] & attempts["_other"]).sum())
    invalid_points = int((~attempts["_ojt"] & attempts["_grade_point_invalid"]).sum())
    missing_points = int((~attempts["_ojt"] & ~attempts["_grade_point_valid"] & ~attempts["_grade_point_invalid"]).sum())
    values = [
        ("Raw transcript rows", raw),
        ("Valid difficulty-attempt rows", int(len(valid))),
        ("OJT rows excluded", ojt),
        ("Foundation rows excluded from modeling table", foundation),
        ("Spring 2026 rows", spring_rows),
        ("Historical rows", historical_rows),
        ("Historical modeling rows", int(len(historical))),
        ("Spring 2026 descriptive rows excluding foundation", int(len(spring))),
        ("Rows with missing course code", missing_code),
        ("Rows with missing result classification", unclassified),
        ("Rows with invalid grade point", invalid_points),
        ("Rows with missing grade point", missing_points),
        ("Courses represented", int(len(difficulty))),
        ("Prepared historical file rows", historical_reference_rows if historical_reference_rows is not None else "Not supplied"),
        ("Prepared Spring 2026 holdout rows", holdout_reference_rows if holdout_reference_rows is not None else "Not supplied"),
        ("Pass rate denominator", "Total Attempts, including passes, failures, withdrawals, and other unclassified attempts"),
        ("Fail rate denominator", "Total Attempts"),
        ("Withdrawal rate denominator", "Total Attempts"),
        ("Repeat rate denominator", "Total Attempts"),
        ("Repeat count definition", "A later attempt classified from the immediately previous attempt: Repeat Due To Failure, Repeat After Withdrawal, Repeat For Improvement, or Repeat Type Unresolved. Stored flags are not overwritten."),
        ("Unresolved repeat definition", "The immediately previous attempt has no usable pass, fail, or withdrawal outcome. These rows are not reclassified from a stored flag."),
        ("Officially Postponed denominator", "Grade OP is described as Officially Postponed. Whether OP stays in the pass-rate denominator is a methodology decision requiring confirmation. The current denominator is unchanged."),
        ("General Requirement pool", "General Requirement Elective — Official Named Pool Required."),
        ("Low sample warning", f"Total Attempts below {LOW_SAMPLE_ATTEMPTS}. This threshold is an analytical reliability flag, not an official academic rule."),
        ("Difficulty rule", "Easy when pass rate is at least 85 percent. Medium when pass rate is at least 65 percent and below 85 percent. Hard when pass rate is below 65 percent."),
        ("Analysis scopes", f"{SCOPE_ALL} / {PURPOSE_ALL}; {SCOPE_HISTORICAL} / {PURPOSE_HISTORICAL}"),
    ]
    return pd.DataFrame(values, columns=["Metric", "Value"])


def _paper_summary(difficulty: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    courses = int(len(difficulty))
    attempts = int(valid.shape[0])
    passed = int(valid["_passed"].sum()) if attempts else 0
    failed = int(valid["_failed"].sum()) if attempts else 0
    withdrawn = int(valid["_withdrawn"].sum()) if attempts else 0
    students = int(valid["_student"].replace("", pd.NA).nunique(dropna=True)) if attempts else 0
    counts = difficulty["Difficulty Level"].value_counts() if courses else pd.Series(dtype=int)

    def share(level: str) -> str:
        count = int(counts.get(level, 0))
        percent = 100 * count / courses if courses else 0
        return f"{count} ({percent:.1f}%)"

    values = [
        ("Number of courses analyzed", courses),
        ("Total attempts", attempts),
        ("Unique students", students),
        ("Overall pass count", passed),
        ("Overall fail count", failed),
        ("Overall withdrawal count", withdrawn),
        ("Overall pass rate", passed / attempts if attempts else None),
        ("Easy courses", share(LEVEL_EASY)),
        ("Medium courses", share(LEVEL_MEDIUM)),
        ("Hard courses", share(LEVEL_HARD)),
        ("Course with highest pass rate", _extreme(difficulty, "Pass Rate", highest=True)),
        ("Course with lowest pass rate", _extreme(difficulty, "Pass Rate", highest=False)),
        ("Course with highest failure rate", _extreme(difficulty, "Fail Rate", highest=True)),
        ("Course with highest withdrawal rate", _extreme(difficulty, "Withdrawal Rate", highest=True)),
        ("Course with highest repeat count", _extreme(difficulty, "Repeat Count", highest=True)),
        ("Difficulty wording", "A course is historically classified as Easy, Medium, or Hard based on the project pass-rate threshold. The label is an observed historical category."),
        ("Analysis scope", SCOPE_ALL),
        ("Analysis purpose", PURPOSE_ALL),
    ]
    return pd.DataFrame(values, columns=["Metric", "Value"])


def _validate(
    attempts: pd.DataFrame,
    valid: pd.DataFrame,
    difficulty: pd.DataFrame,
    features: pd.DataFrame,
    historical: pd.DataFrame,
    catalog: dict[str, dict[str, object]],
    pool_codes: set[str],
    *,
    historical_reference_rows: int | None,
    holdout_reference_rows: int | None,
) -> pd.DataFrame:
    raw = int(len(attempts))
    ojt = int(attempts["_ojt"].sum())
    spring = int(attempts["_spring_2026"].sum())
    reconciled = int(len(valid)) + ojt == raw
    if historical_reference_rows is not None and holdout_reference_rows is not None:
        reconciled = reconciled and historical_reference_rows + holdout_reference_rows == raw and historical_reference_rows == raw - spring
    attempt_sum = int(difficulty["Total Attempts"].sum()) if not difficulty.empty else 0
    overlap = int((attempts["_passed"] & attempts["_failed"]).sum() + (attempts["_passed"] & attempts["_withdrawn"]).sum() + (attempts["_failed"] & attempts["_withdrawn"]).sum())
    unknown = int(difficulty["Course Reference Category"].eq(CATEGORY_UNKNOWN).sum()) if not difficulty.empty and "Course Reference Category" in difficulty.columns else 0
    if pool_codes and not difficulty.empty:
        elective_marked_unknown = int(difficulty.loc[difficulty["Course Code"].isin(pool_codes), "Course Reference Category"].eq(CATEGORY_UNKNOWN).sum())
    else:
        elective_marked_unknown = 0
    low_sample = int(difficulty["Low Sample Warning"].eq(WARNING_YES).sum()) if not difficulty.empty else 0
    spring_in_features = int(historical["_spring_2026"].sum()) if not historical.empty else 0
    ojt_codes = set(attempts.loc[attempts["_ojt"], "_code"]) - {""}
    foundation_codes = set(attempts.loc[attempts["_foundation"], "_code"]) - {""}
    feature_codes = set(features["Course Code"]) if not features.empty else set()
    duplicate_difficulty = int(difficulty["Course Code"].duplicated().sum()) if not difficulty.empty else 0
    duplicate_features = int(features["Course Code"].duplicated().sum()) if not features.empty else 0
    checks = [
        _check("Total transcript attempt rows reconciled", int(len(valid)) + ojt, fail=not reconciled),
        _check("Duplicate attempts retained", attempt_sum, fail=attempt_sum != int(len(valid))),
        _check("Course codes missing", int(attempts["_code"].eq("").sum()), fail=int(attempts["_code"].eq("").sum()) > 0),
        _check("Course titles missing", int(difficulty["Course Title"].map(_text).eq("").sum()) if not difficulty.empty else 0, fail=bool(not difficulty.empty and difficulty["Course Title"].map(_text).eq("").any())),
        _check("Unknown course codes", unknown, review=unknown > 0),
        _check("Elective pool courses classified as unknown", elective_marked_unknown, fail=elective_marked_unknown > 0),
        _check("Pass/fail/withdraw flags mutually interpretable", overlap, fail=overlap > 0),
        _check("Pass count greater than attempts", _exceeds(difficulty, "Pass Count"), fail=_exceeds(difficulty, "Pass Count") > 0),
        _check("Fail count greater than attempts", _exceeds(difficulty, "Fail Count"), fail=_exceeds(difficulty, "Fail Count") > 0),
        _check("Withdraw count greater than attempts", _exceeds(difficulty, "Withdrawn Count"), fail=_exceeds(difficulty, "Withdrawn Count") > 0),
        _check("Pass rate outside 0-1", _outside_rate(difficulty, "Pass Rate"), fail=_outside_rate(difficulty, "Pass Rate") > 0),
        _check("Fail rate outside 0-1", _outside_rate(difficulty, "Fail Rate"), fail=_outside_rate(difficulty, "Fail Rate") > 0),
        _check("Withdrawal rate outside 0-1", _outside_rate(difficulty, "Withdrawal Rate"), fail=_outside_rate(difficulty, "Withdrawal Rate") > 0),
        _check("Difficulty category outside Easy/Medium/Hard", _bad_level(difficulty), fail=_bad_level(difficulty) > 0),
        _check("Easy course with pass rate < 0.85", _bad_easy(difficulty), fail=_bad_easy(difficulty) > 0),
        _check("Medium course with pass rate < 0.65 or >= 0.85", _bad_medium(difficulty), fail=_bad_medium(difficulty) > 0),
        _check("Hard course with pass rate >= 0.65", _bad_hard(difficulty), fail=_bad_hard(difficulty) > 0),
        _check("Average grade point outside valid scale", _bad_average(difficulty), fail=_bad_average(difficulty) > 0),
        _check("Spring 2026 rows in historical modeling features", spring_in_features, fail=spring_in_features > 0),
        _check("OJT courses in modeling difficulty table", int(len(feature_codes & ojt_codes)), fail=bool(feature_codes & ojt_codes)),
        _check("Foundation courses in modeling difficulty table", int(len(feature_codes & foundation_codes)), fail=bool(feature_codes & foundation_codes)),
        _check("Duplicate Course Code rows in one-scope difficulty table", duplicate_difficulty + duplicate_features, fail=duplicate_difficulty + duplicate_features > 0),
        _check("Outcome counts do not sum to attempts", _outcome_gap(difficulty), fail=_outcome_gap(difficulty) > 0),
        _check("Repeat count greater than attempts", _exceeds(difficulty, "Repeat Count"), fail=_exceeds(difficulty, "Repeat Count") > 0),
        _check("Low sample courses", low_sample, review=low_sample > 0),
    ]
    return pd.DataFrame(checks)


def _official_pools(pools: pd.DataFrame | None) -> pd.DataFrame:
    if pools is not None:
        return pools
    from project_paths import REFERENCE_WORKBOOK

    if REFERENCE_WORKBOOK.exists():
        return pd.read_excel(REFERENCE_WORKBOOK, sheet_name="Elective Pools")
    return pd.DataFrame(columns=["Elective Pool", "Course Code"])


def _pool_codes(pools: pd.DataFrame) -> set[str]:
    if pools is None or pools.empty or "Course Code" not in pools.columns:
        return set()
    return {code for code in pools["Course Code"].map(_code) if code}


def _reference_category(code: str, in_plan: bool, in_pool: bool) -> str:
    if not code:
        return CATEGORY_UNKNOWN
    if in_plan:
        return CATEGORY_PLAN
    if in_pool:
        return CATEGORY_POOL
    if code in GENERAL_REQUIREMENT_GAP:
        return CATEGORY_GENERAL
    return CATEGORY_HISTORICAL


def _outcome_audit(attempts: pd.DataFrame) -> pd.DataFrame:
    columns = ["Student Code", "Course Code", "Semester", "Grade", "Grade Point", "Result", "Derived Academic Outcome", "Methodology Note"]
    if attempts.empty or "_derived_outcome" not in attempts.columns:
        return pd.DataFrame(columns=columns)
    special = attempts.loc[attempts["_derived_outcome"].isin([OUTCOME_FAILED_RECOVERED, OUTCOME_POSTPONED, OUTCOME_MANUAL])].copy()
    rows = []
    for _, row in special.iterrows():
        outcome = row["_derived_outcome"]
        if outcome == OUTCOME_POSTPONED:
            note = "Grade OP follows the transcript legend: Officially Postponed. Inclusion in the difficulty denominator is not decided here."
        elif outcome == OUTCOME_FAILED_RECOVERED:
            note = "Raw Result stays corrupted for audit. Grade F, grade point 0, and Is Failed supply the downstream outcome."
        else:
            note = "Manual Review. Grade and Result cannot be reconstructed from the current project evidence."
        rows.append({
            "Student Code": row.get("Student Code"),
            "Course Code": row.get("Course Code"),
            "Semester": row.get("Semester"),
            "Grade": row.get("Grade"),
            "Grade Point": row.get("Grade Point"),
            "Result": row.get("Result"),
            "Derived Academic Outcome": outcome,
            "Methodology Note": note,
        })
    return _frame(rows, columns, ["Student Code", "Course Code"])


def _mismatch_audit(attempts: pd.DataFrame) -> pd.DataFrame:
    columns = ["Student Code", "Course Code", "Semester", "Attempt Number", "Derived Repeat Type", "Is Repeated Due To Failure", "Is Repeated For Improvement", "Repeat Flag Mismatch"]
    if attempts.empty or "_repeat_mismatch" not in attempts.columns:
        return pd.DataFrame(columns=columns)
    flagged = attempts.loc[attempts["_repeat_mismatch"].map(_text).ne("")].copy()
    rows = []
    for _, row in flagged.iterrows():
        rows.append({
            "Student Code": row.get("Student Code"),
            "Course Code": row.get("Course Code"),
            "Semester": row.get("Semester"),
            "Attempt Number": row.get("Attempt Number"),
            "Derived Repeat Type": row.get("_derived_repeat"),
            "Is Repeated Due To Failure": row.get("Is Repeated Due To Failure"),
            "Is Repeated For Improvement": row.get("Is Repeated For Improvement"),
            "Repeat Flag Mismatch": row.get("_repeat_mismatch"),
        })
    return _frame(rows, columns, ["Student Code", "Course Code", "Attempt Number"])


def _plan_catalog(plan: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, set[str] | set[float]]] = {}
    if plan is None or plan.empty or "Course Code" not in plan.columns:
        return {}
    for _, row in plan.iterrows():
        code = _code(row.get("Course Code"))
        if not code:
            continue
        item = catalog.setdefault(code, {"titles": set(), "paths": set(), "levels": set(), "years": set(), "types": set(), "requirements": set(), "credits": set()})
        _add_text(item["titles"], row.get("Course Title"))
        _add_text(item["paths"], row.get("Specialization Path"))
        _add_text(item["levels"], row.get("Level"))
        _add_text(item["years"], row.get("Year"))
        _add_text(item["types"], row.get("Course Type"))
        _add_text(item["requirements"], row.get("Requirement Type"))
        credit = _number(row.get("Credit Hours"))
        if credit is not None:
            item["credits"].add(credit)
    return {
        code: {
            "title": _join(item["titles"]) if len(item["titles"]) == 1 else _join(item["titles"]),
            "paths": _join(item["paths"]),
            "level": _join(item["levels"]),
            "year": _join(item["years"]),
            "course_type": _join(item["types"]),
            "requirement_type": _join(item["requirements"]),
            "credits": next(iter(item["credits"])) if len(item["credits"]) == 1 else None,
        }
        for code, item in catalog.items()
    }


def _groups(frame: pd.DataFrame):
    if frame.empty:
        return []
    usable = frame.loc[frame["_code"].ne("")].copy()
    if usable.empty:
        return []
    return list(usable.groupby("_code", sort=True))


def _spring_mask(frame: pd.DataFrame) -> pd.Series:
    semester = frame["Semester"].map(_text) if "Semester" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    year = frame["Academic Year"].map(_text) if "Academic Year" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    term = frame["Term"].map(_text) if "Term" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    return semester.str.contains(HOLDOUT_TOKEN, na=False) | year.eq(HOLDOUT_TOKEN) | (year.str.contains(HOLDOUT_TOKEN, na=False) & term.str.contains("Spring", case=False, na=False))


def _is_ojt(code: str, name: str) -> bool:
    if code and code_ojt_status(code) == "confirmed":
        return True
    return title_ojt_status(normalize_course_title(name)) == "confirmed"


def _evidence_quality(attempts: int) -> str:
    if attempts < LOW_SAMPLE_ATTEMPTS:
        return "Low Evidence"
    if attempts < MODERATE_SAMPLE_ATTEMPTS:
        return "Moderate Evidence"
    return "High Evidence"


def _extreme(frame: pd.DataFrame, column: str, *, highest: bool) -> str:
    if frame.empty or frame[column].dropna().empty:
        return ""
    value = frame[column].max() if highest else frame[column].min()
    matched = frame.loc[frame[column].eq(value), "Course Code"].map(_text).sort_values()
    codes = " | ".join(code for code in matched if code)
    return f"{codes} ({value:.4f})" if codes else ""


def _exceeds(frame: pd.DataFrame, column: str) -> int:
    if frame.empty:
        return 0
    return int((frame[column] > frame["Total Attempts"]).sum())


def _outside_rate(frame: pd.DataFrame, column: str) -> int:
    if frame.empty:
        return 0
    values = frame[column].map(_number)
    return int((values.notna() & ((values < 0) | (values > 1))).sum())


def _bad_level(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int((~frame["Difficulty Level"].isin([LEVEL_EASY, LEVEL_MEDIUM, LEVEL_HARD])).sum())


def _bad_easy(frame: pd.DataFrame) -> int:
    return _threshold_mismatch(frame, LEVEL_EASY)


def _bad_medium(frame: pd.DataFrame) -> int:
    return _threshold_mismatch(frame, LEVEL_MEDIUM)


def _bad_hard(frame: pd.DataFrame) -> int:
    return _threshold_mismatch(frame, LEVEL_HARD)


def _threshold_mismatch(frame: pd.DataFrame, level: str) -> int:
    if frame.empty:
        return 0
    mismatches = 0
    for _, row in frame.iterrows():
        expected = difficulty_level(int(row["Pass Count"]), int(row["Total Attempts"]))
        if row["Difficulty Level"] == level and expected != level:
            mismatches += 1
    return mismatches


def _bad_average(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    values = frame["Average Grade Point"].map(_number)
    return int((values.notna() & ((values < GRADE_POINT_MINIMUM) | (values > GRADE_POINT_MAXIMUM))).sum())


def _outcome_gap(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    total = frame["Pass Count"] + frame["Fail Count"] + frame["Withdrawn Count"] + frame["Other / Unclassified Attempt Count"]
    return int((total != frame["Total Attempts"]).sum())


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _single_number(values: pd.Series) -> float | None:
    numbers = {value for value in values.map(_number) if value is not None}
    if len(numbers) == 1:
        return next(iter(numbers))
    return None


def _add_text(bucket: set[str], value: object) -> None:
    text = _text(value)
    if text:
        bucket.add(text)


def _join(values: set[str]) -> str:
    return " | ".join(sorted(values))


def _figure_pass_rate_histogram(frame, directory, data_directory, plt) -> list[Path]:
    rates = frame["Pass Rate"].map(_number).dropna()
    rates.to_frame("Pass Rate").to_csv(data_directory / "fig01_pass_rate_distribution.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(rates, bins=10, range=(0, 1), color="#1D3557", edgecolor="white")
    axis.set_title("Course pass-rate distribution")
    axis.set_xlabel("Pass rate")
    axis.set_ylabel("Number of courses")
    axis.set_xlim(0, 1)
    return _save(figure, directory, "fig01_pass_rate_distribution", plt)


def _figure_level_counts(frame, directory, data_directory, plt) -> list[Path]:
    order = [LEVEL_EASY, LEVEL_MEDIUM, LEVEL_HARD]
    counts = frame["Difficulty Level"].value_counts().reindex(order, fill_value=0)
    counts.rename_axis("Difficulty Level").reset_index(name="Courses").to_csv(data_directory / "fig02_difficulty_level_counts.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(order, counts.values, color=["#2A9D8F", "#E9C46A", "#E76F51"])
    axis.set_title("Courses by official pass-rate difficulty level")
    axis.set_xlabel("Difficulty level")
    axis.set_ylabel("Number of courses")
    for bar, count in zip(bars, counts.values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha="center", va="bottom")
    return _save(figure, directory, "fig02_difficulty_level_counts", plt)


def _figure_course_bars(frame, directory, data_directory, plt) -> list[Path]:
    ordered = frame.sort_values(["Pass Rate", "Course Code"], kind="mergesort")
    ordered.loc[:, ["Course Code", "Course Title", "Pass Rate", "Difficulty Level", "Total Attempts", "Low Sample Warning"]].to_csv(data_directory / "fig03_course_pass_rates.csv", index=False)
    colors = ordered["Difficulty Level"].map({LEVEL_EASY: "#2A9D8F", LEVEL_MEDIUM: "#E9C46A", LEVEL_HARD: "#E76F51"}).fillna("#6C757D")
    figure, axis = plt.subplots(figsize=(10, max(6, 0.28 * max(len(ordered), 1))))
    axis.barh(ordered["Course Code"], ordered["Pass Rate"], color=colors)
    axis.set_title("Course pass rates, sorted from lowest to highest")
    axis.set_xlabel("Pass rate")
    axis.set_ylabel("Course code")
    axis.set_xlim(0, 1)
    figure.tight_layout()
    return _save(figure, directory, "fig03_course_pass_rates", plt)


def _figure_grade_scatter(frame, directory, data_directory, plt) -> list[Path]:
    usable = frame.loc[frame["Average Grade Point"].map(_number).notna()].copy()
    usable.loc[:, ["Course Code", "Average Grade Point", "Pass Rate", "Difficulty Level"]].to_csv(data_directory / "fig04_grade_point_vs_pass_rate.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(usable["Average Grade Point"], usable["Pass Rate"], c="#1D3557", alpha=0.8)
    _label_extremes(axis, usable, "Average Grade Point", "Pass Rate")
    axis.set_title("Average grade point and pass rate")
    axis.set_xlabel("Average grade point")
    axis.set_ylabel("Pass rate")
    axis.set_ylim(0, 1.05)
    return _save(figure, directory, "fig04_grade_point_vs_pass_rate", plt)


def _figure_repeat_scatter(frame, directory, data_directory, plt) -> list[Path]:
    plotting = frame.copy()
    plotting["Repeat Rate"] = plotting["Repeat Count"] / plotting["Total Attempts"].replace(0, pd.NA)
    plotting.loc[:, ["Course Code", "Fail Rate", "Repeat Rate", "Repeat Count", "Total Attempts"]].to_csv(data_directory / "fig05_failure_vs_repeat.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(plotting["Fail Rate"], plotting["Repeat Rate"], c="#9B2226", alpha=0.8)
    axis.set_title("Failure rate and repeat rate")
    axis.set_xlabel("Fail rate")
    axis.set_ylabel("Repeat rate")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    return _save(figure, directory, "fig05_failure_vs_repeat", plt)


def _figure_withdrawal_bars(frame, directory, data_directory, plt) -> list[Path]:
    observed = frame.loc[frame["Withdrawn Count"] > 0].sort_values(["Withdrawal Rate", "Course Code"], kind="mergesort")
    observed.loc[:, ["Course Code", "Withdrawal Rate", "Withdrawn Count", "Total Attempts", "Low Sample Warning"]].to_csv(data_directory / "fig06_withdrawal_rates.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, max(5, 0.32 * max(len(observed), 1))))
    if observed.empty:
        axis.text(0.5, 0.5, "No withdrawn attempts", ha="center", va="center")
    else:
        colors = observed["Low Sample Warning"].map({WARNING_YES: "#F4A261", WARNING_NO: "#1D3557"}).fillna("#1D3557")
        axis.barh(observed["Course Code"], observed["Withdrawal Rate"], color=colors)
    axis.set_title("Withdrawal rate for courses with at least one withdrawal")
    axis.set_xlabel("Withdrawal rate")
    axis.set_ylabel("Course code")
    axis.set_xlim(0, 1)
    figure.tight_layout()
    return _save(figure, directory, "fig06_withdrawal_rates", plt)


def _label_extremes(axis, frame: pd.DataFrame, x_column: str, y_column: str) -> None:
    if frame.empty:
        return
    lowest = frame.sort_values([y_column, "Course Code"], kind="mergesort").iloc[0]
    highest = frame.sort_values([y_column, "Course Code"], ascending=[False, True], kind="mergesort").iloc[0]
    for row in (lowest, highest):
        axis.annotate(row["Course Code"], (row[x_column], row[y_column]), textcoords="offset points", xytext=(4, 4), fontsize=8)


def _save(figure, directory: Path, stem: str, plt) -> list[Path]:
    figure.tight_layout()
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _frame(rows: list[dict[str, object]], columns: list[str], sort_by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(sort_by, kind="mergesort").reset_index(drop=True)


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


def _code(value: object) -> str:
    normalized = normalize_course_code(value)
    try:
        if pd.isna(normalized):
            return ""
    except TypeError:
        if normalized is None:
            return ""
    return str(normalized)


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


def _grade_point(value: object) -> float | None:
    return _number(value)


def _invalid_grade_point(value: object) -> bool:
    number = _number(value)
    return number is not None and (number < GRADE_POINT_MINIMUM or number > GRADE_POINT_MAXIMUM)


def _attempt_value(value: object) -> int:
    number = _number(value)
    if number is None:
        return 1
    return int(number)


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
