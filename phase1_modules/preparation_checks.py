"""Shared checks for period, numeric values, stored flags, and course identity.

The notebook calls these functions so preparation and exploratory analysis use
the same rules. The functions do not modify source workbooks.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

TERM_SEQUENCE: dict[str, int] = {"spring": 0, "fall": 1}
TARGET_YEAR = 2026
TARGET_TERM_KEY = "spring"
TARGET_ORDER = TARGET_YEAR * 2 + TERM_SEQUENCE[TARGET_TERM_KEY]

OJT_TITLE_PATTERN = re.compile(r"\bon\s+(?:the\s+)?job\s+training\b|\bojt\b")
OJT_CODE_PATTERN = re.compile(r"(?<![A-Z])OJT(?![A-Z])")

_TRUE_TEXT = {"true", "yes", "y", "1"}
_FALSE_TEXT = {"false", "no", "n", "0"}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write to a temporary file and replace the target only after success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        write(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def classify_numeric_value(
    value: object,
    *,
    minimum: float | None = 0,
    maximum: float | None = None,
) -> str:
    """Classify one stored number into exactly one category.

    Categories are mutually exclusive: blank, nonnumeric, nonfinite, negative,
    below_minimum, above_maximum, or ok. Positive infinity is nonfinite, not
    above the maximum.
    """
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return "blank"
        try:
            number = float(text)
        except ValueError:
            return "nonnumeric"
    elif _is_missing(value) or isinstance(value, bool):
        return "blank" if _is_missing(value) else "nonnumeric"
    elif isinstance(value, (int, np.integer, float, np.floating)):
        number = float(value)
    else:
        return "nonnumeric"
    if not math.isfinite(number):
        return "nonfinite"
    if minimum is not None and number < minimum:
        return "negative" if minimum == 0 else "below_minimum"
    if maximum is not None and number > maximum:
        return "above_maximum"
    return "ok"


def audit_numeric_series(
    values: pd.Series,
    *,
    minimum: float | None = 0,
    maximum: float | None = None,
) -> pd.Series:
    """Return one category for every value, using the same classifier as the detail rows."""
    return values.map(lambda value: classify_numeric_value(value, minimum=minimum, maximum=maximum))


def interpret_flag(value: object) -> bool | None:
    """Return True, False, or None for an unrecognized or missing flag.

    Documented representations are booleans, 1, 0, and the words yes/no and
    true/false. Other strings are not forced through bool().
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    if isinstance(value, str):
        key = value.strip().casefold()
        if key in _TRUE_TEXT:
            return True
        if key in _FALSE_TEXT:
            return False
        return None
    return None


def normalize_flag_series(values: pd.Series) -> pd.Series:
    """Map stored flags to nullable Boolean values.

    Documented true and false representations become True and False.
    Missing and unrecognized values stay missing. They are not stored as False.
    The original series is not changed, and the result does not decide what the
    flag means academically.
    """
    return values.map(interpret_flag).astype("boolean")


def known_false_mask(values: pd.Series) -> pd.Series:
    """Return True only where the normalized flag is False.

    Missing and unrecognized values are left out of the mask. They are not
    treated as False.
    """
    normalized = normalize_flag_series(values)
    return normalized.eq(False).fillna(False).astype(bool)


def aggregate_normalized_flags(
    frame: pd.DataFrame,
    keys: list[str],
    flag_columns: list[str],
) -> pd.DataFrame:
    """Count documented true flags within each group.

    Each flag produces ``{column} true`` and ``{column} unknown``. Unknown
    values are omitted from the true count. That count is an exact group total
    only when the unknown count is zero. Source columns are not modified.
    """
    work = frame.loc[:, list(keys)].copy()
    for column in flag_columns:
        normalized = normalize_flag_series(frame[column])
        work[f"{column} true"] = normalized.eq(True).fillna(False).astype("int64")
        work[f"{column} unknown"] = normalized.isna().astype("int64")
    value_columns = [column for column in work.columns if column not in keys]
    return (
        work.groupby(list(keys), dropna=True)[value_columns]
        .sum()
        .reset_index()
    )


def compare_flag_aggregate_series(
    stored: pd.Series,
    true_count: pd.Series,
    unknown_count: pd.Series,
) -> pd.Series:
    """Label each aggregate exact, mismatch, or incomplete.

    An unknown flag makes the true count incomplete. It is not compared as an
    exact total and it is not converted to false.
    """
    unknown_number = pd.to_numeric(unknown_count, errors="coerce").fillna(0)
    mismatch = unknown_number.eq(0) & stored.fillna(-1).ne(true_count.fillna(-1))
    status = pd.Series("exact", index=stored.index, dtype="object")
    status = status.mask(unknown_number.gt(0), "incomplete")
    status = status.mask(mismatch, "mismatch")
    return status


def page_retrieval_date(filename: str, fetched_now: set[str], today: str) -> str:
    """Return a retrieval date only for a page fetched in this run.

    Cached pages, and pages that were not fetched successfully, stay
    ``Not recorded``. A file modification time is not a retrieval date.
    """
    if filename in fetched_now:
        return today
    return "Not recorded"


def fetch_description_pages(
    urls: list[str],
    destination_dir: Path,
    fetch: Callable[[str], bytes],
    filename_for: Callable[[str], str],
) -> tuple[set[str], list[str]]:
    """Save successful downloads and list the failures.

    Only a page whose fetch returns bytes is written and included in the
    fetched set. A failed fetch does not delete an existing file.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fetched: set[str] = set()
    failures: list[str] = []
    for url in urls:
        try:
            payload = fetch(url)
        except Exception as error:
            failures.append(f"{url}: {type(error).__name__}: {error}")
            continue
        name = filename_for(url)
        (destination / name).write_bytes(payload)
        fetched.add(name)
    return fetched, failures


def _classify_year(value: object) -> tuple[str, int | None]:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return "missing", None
        try:
            number = float(text)
        except ValueError:
            return "nonnumeric", None
    elif isinstance(value, bool):
        return "nonnumeric", None
    elif _is_missing(value):
        return "missing", None
    elif isinstance(value, (int, np.integer)):
        return "ok", int(value)
    elif isinstance(value, (float, np.floating)):
        number = float(value)
    else:
        return "nonnumeric", None
    if not math.isfinite(number):
        return "nonfinite", None
    if not float(number).is_integer():
        return "not_integer", None
    return "ok", int(number)


def _classify_term(value: object) -> tuple[str, str | None]:
    if _is_missing(value):
        return "missing", None
    key = str(value).strip().casefold()
    if key == "":
        return "missing", None
    if key in TERM_SEQUENCE:
        return "ok", key
    return "unrecognized", None


def _semester_key(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text or None


def classify_period_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify Academic Year, Term, and Semester without raising on invalid values.

    A year must be a finite whole number before it receives a period order.
    Fractional and infinite years stay unresolved, so they cannot share the
    Spring 2026 order. The input frame is not modified.
    """
    required = ["Academic Year", "Term", "Semester"]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError("Period classification needs these columns: " + ", ".join(missing_columns))
    work = frame.copy()
    orders: list[int | None] = []
    consistent: list[bool] = []
    partitions: list[str] = []
    reasons: list[str | None] = []
    labels: list[str | None] = []
    validated_years: list[int | None] = []
    for year_value, term_value, semester_value in zip(
        work["Academic Year"], work["Term"], work["Semester"]
    ):
        year_status, year_number = _classify_year(year_value)
        term_status, term_key = _classify_term(term_value)
        semester = _semester_key(semester_value)
        reason = None
        order = None
        label = None
        is_consistent = False
        if year_status == "missing" or term_status == "missing" or semester is None:
            reason = "Missing academic-period value"
        elif year_status == "nonfinite":
            reason = "Academic year is not finite"
        elif year_status == "not_integer":
            reason = "Academic year is not a whole number"
        elif year_status == "nonnumeric":
            reason = "Academic year is not numeric"
        elif term_status == "unrecognized":
            reason = "Unrecognized term"
        else:
            assert year_number is not None and term_key is not None
            expected = f"{year_number} {term_key}"
            label = f"{year_number} {str(term_value).strip()}"
            if semester != expected:
                reason = "Semester text conflicts with Academic Year and Term"
            else:
                is_consistent = True
                order = year_number * 2 + TERM_SEQUENCE[term_key]
                if order > TARGET_ORDER:
                    reason = "Period is later than Spring 2026"
        if not is_consistent:
            partition = "Unresolved"
        elif order is not None and order < TARGET_ORDER:
            partition = "Before Spring 2026"
        elif order == TARGET_ORDER:
            partition = "Spring 2026"
        else:
            partition = "After Spring 2026"
        orders.append(order)
        consistent.append(is_consistent)
        partitions.append(partition)
        reasons.append(reason)
        labels.append(label)
        validated_years.append(year_number if is_consistent else None)
    work["Validated academic year"] = pd.Series(validated_years, index=work.index, dtype="Int64")
    work["Period Order"] = pd.Series(orders, index=work.index, dtype="Int64")
    work["Period Consistent"] = pd.Series(consistent, index=work.index, dtype="boolean")
    work["Period Label"] = pd.Series(labels, index=work.index, dtype="string")
    work["Partition"] = pd.Series(partitions, index=work.index, dtype="string")
    work["Review Reason"] = pd.Series(reasons, index=work.index, dtype="string")
    return work


def normalize_course_code(value: object) -> object:
    """Uppercase a code and remove separators. Missing values stay missing."""
    if _is_missing(value):
        return pd.NA
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).strip()).upper()
    return text if text else pd.NA


def normalize_course_title(value: object) -> str:
    """Lowercase a title and treat hyphens as spaces. Used only for matching."""
    if _is_missing(value):
        return ""
    text = str(value).casefold().replace("–", "-").replace("—", "-")
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_ojt_status(title_key: str) -> str:
    """Return 'confirmed' when the normalized title is an explicit OJT phrase."""
    if title_key and OJT_TITLE_PATTERN.search(title_key):
        return "confirmed"
    return ""


def ojt_title_phrase(title_key: str) -> str:
    """Name the explicit phrase that confirmed an OJT title."""
    if re.search(r"\bon\s+the\s+job\s+training\b", title_key):
        return "on the job training"
    if re.search(r"\bon\s+job\s+training\b", title_key):
        return "on job training"
    if re.search(r"\bojt\b", title_key):
        return "ojt"
    return ""


def code_ojt_status(code_key: object) -> str:
    """Return 'confirmed' when the normalized code contains the token OJT."""
    if _is_missing(code_key):
        return ""
    if OJT_CODE_PATTERN.search(str(code_key)):
        return "confirmed"
    return ""


def is_foundation_code(code_key: object) -> bool:
    """Return whether the row's own course code begins with FP."""
    return (not _is_missing(code_key)) and str(code_key).startswith("FP")


PASSING_GRADE_COLUMNS = ("Passing Grade", "Passing Grade Point")
REMAINING_PASSING_GRADE = "C"


def passing_grade_table(source: pd.DataFrame) -> pd.DataFrame:
    """Return one passing-grade row per course code from the supplied workbook.

    The source workbook is not modified. A repeated course code is rejected.
    """
    required = ["Course Code", "Passing Grade", "Passing Grade Point"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Passing grades workbook is missing columns: " + ", ".join(missing))
    table = source.loc[:, required].copy()
    table["Course Code"] = table["Course Code"].map(normalize_course_code)
    table["Passing Grade"] = table["Passing Grade"].astype("string").str.strip()
    table["Passing Grade Point"] = pd.to_numeric(table["Passing Grade Point"], errors="coerce")
    if table["Course Code"].isna().any() or table["Passing Grade"].isna().any() or table["Passing Grade"].eq("").any():
        raise ValueError("Passing grades.xlsx has a blank course code or passing grade.")
    if table["Passing Grade Point"].isna().any() or not table["Passing Grade Point"].map(math.isfinite).all():
        raise ValueError("Passing grades.xlsx has a missing or nonfinite passing grade point.")
    if table["Course Code"].duplicated().any():
        repeated = table.loc[table["Course Code"].duplicated(), "Course Code"].astype(str).unique()
        raise ValueError("Passing grades.xlsx repeats course codes: " + ", ".join(repeated))
    c_points = table.loc[table["Passing Grade"].eq(REMAINING_PASSING_GRADE), "Passing Grade Point"]
    if c_points.empty or c_points.nunique() != 1:
        raise ValueError("Passing grades.xlsx must state one grade point for C.")
    return table.reset_index(drop=True)


def c_grade_point(rules: pd.DataFrame) -> float:
    """Return the grade point recorded for C in the passing-grade table."""
    points = rules.loc[rules["Passing Grade"].eq(REMAINING_PASSING_GRADE), "Passing Grade Point"]
    return float(points.iloc[0])


def attach_passing_grades(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    code_column: str = "Course Code",
    elective_column: str | None = None,
    all_elective: bool = False,
) -> pd.DataFrame:
    """Add the minimum passing grade without changing existing rows or values.

    A course listed in the passing-grade table keeps that grade. Advanced
    Diploma, Bachelor, and any other course absent from the table use C.
    Elective rows use C even when a listed grade is present.
    """
    work = frame.copy()
    codes = work[code_column].map(normalize_course_code)
    listed_grade = codes.map(rules.set_index("Course Code")["Passing Grade"])
    listed_point = codes.map(rules.set_index("Course Code")["Passing Grade Point"])
    if all_elective:
        elective = pd.Series(True, index=work.index)
    elif elective_column:
        elective = work[elective_column].astype("string").str.strip().str.casefold().eq("yes")
    else:
        elective = pd.Series(False, index=work.index)
    minimum_point = c_grade_point(rules)
    grade = listed_grade.astype("string")
    point = listed_point.astype("float64")
    grade = grade.mask(listed_grade.isna(), REMAINING_PASSING_GRADE)
    point = point.mask(listed_point.isna(), minimum_point)
    grade = grade.mask(elective, REMAINING_PASSING_GRADE)
    point = point.mask(elective, minimum_point)
    work["Passing Grade"] = grade.astype("string")
    work["Passing Grade Point"] = point.astype("float64")
    if len(work) != len(frame) or list(work.columns) != list(frame.columns) + list(PASSING_GRADE_COLUMNS):
        raise RuntimeError("Attaching passing grades changed the original columns or row count.")
    original = [column for column in frame.columns]
    if not work.loc[:, original].equals(frame.loc[:, original]):
        raise RuntimeError("Attaching passing grades changed an existing value.")
    return work


def assign_lookup(frame: pd.DataFrame, key_column: str, lookup: pd.Series) -> pd.Series:
    """Map one unique lookup onto a frame without adding or dropping rows."""
    if lookup.index.duplicated().any():
        raise ValueError("Lookup index is not unique.")
    mapped = frame[key_column].map(lookup)
    if len(mapped) != len(frame):
        raise ValueError("Lookup changed the number of rows.")
    return mapped
