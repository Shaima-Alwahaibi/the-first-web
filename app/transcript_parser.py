"""Turn a machine-readable UTAS transcript PDF into structured attempts.

This module does not apply passing rules, prerequisites, or recommendation scores.
Every extracted attempt is kept, including repeats and Spring 2026 holdout rows.
"""

from __future__ import annotations

import re
from io import BytesIO

import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import HOLD_OUT_LABEL, MSG_IMAGE, MSG_UNREADABLE

GRADE_TOKENS = {
    "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D",
    "F", "FW", "W", "NC", "OP", "IC", "IP", "N", "TC", "CC",
}
RESULT_TOKENS = {"P", "F", "W", "NP"}
COURSE_LINE = re.compile(r"^([A-Z]{3,5}\d{4})\s+(.+)$")
SEMESTER_LINE = re.compile(r"^Semester\s*:\s*(\d{4})\s+(Spring|Summer|Fall|Autumn)\s*$", re.IGNORECASE)
STUDENT_LINE = re.compile(r"Student Code\s*:\s*(STUD-\d+)", re.IGNORECASE)
DEPARTMENT_LINE = re.compile(r"Department\s*:\s*(.+)", re.IGNORECASE)
SPECIALIZATION_LINE = re.compile(r"Specialization\s*:\s*(.+)", re.IGNORECASE)
SUMMARY_LINE = re.compile(
    r"Credits Total Attempted:\s*(\d*)\s+Earned:\s*(\d*)\s+Attempted:\s*(\d*)\s+"
    r"SGPA:\s*([0-9.]*)\s+LCGPA:\s*([0-9.]*)\s+CGPA:\s*([0-9.]*)",
    re.IGNORECASE,
)


class TranscriptReadError(Exception):
    """The PDF could not be read as a supported transcript."""


class ImageTranscriptError(TranscriptReadError):
    """The PDF has no usable text layer."""


def parse_transcript_pdf(payload: bytes, source_name: str = "transcript.pdf") -> dict[str, object]:
    """Extract student details, course attempts, and semester totals from a PDF."""

    text = _extract_text(payload)
    parsed = _parse_text(text)
    parsed["source_name"] = source_name
    student = parsed["student"]
    if isinstance(student, dict):
        student["source_name"] = source_name
    return parsed


def _extract_text(payload: bytes) -> str:
    if not payload:
        raise TranscriptReadError(MSG_UNREADABLE)
    if not payload.startswith(b"%PDF"):
        raise TranscriptReadError(MSG_UNREADABLE)
    try:
        reader = PdfReader(BytesIO(payload))
    except PdfReadError as exc:
        raise TranscriptReadError(MSG_UNREADABLE) from exc
    parts: list[str] = []
    try:
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    except Exception as exc:
        raise TranscriptReadError(MSG_UNREADABLE) from exc
    text = "\n".join(parts).replace("\u00a0", " ").replace("\r", "\n")
    if len(re.sub(r"\s+", "", text)) < 40:
        raise ImageTranscriptError(MSG_IMAGE)
    if "Course No" not in text and not STUDENT_LINE.search(text):
        raise TranscriptReadError(MSG_UNREADABLE)
    return text


def _parse_text(text: str) -> dict[str, object]:
    warnings: list[str] = []
    student_codes = list(dict.fromkeys(STUDENT_LINE.findall(text)))
    department = _first_group(DEPARTMENT_LINE, text)
    specialization = _first_group(SPECIALIZATION_LINE, text)
    if len(student_codes) > 1:
        warnings.append("More than one student code was printed on the transcript. The first code is used.")
    student_code = student_codes[0] if student_codes else ""
    if not student_code:
        warnings.append("Student code was not found on the transcript.")
    if not department:
        warnings.append("Department was not found on the transcript.")
    if not specialization:
        warnings.append("Specialization was not found on the transcript.")

    attempts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    semester = ""
    table_open = False
    table_closed = False
    semester_known_for_table = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        semester_match = SEMESTER_LINE.match(line)
        if semester_match:
            semester = f"{semester_match.group(1)} {semester_match.group(2).title()}"
            table_open = False
            table_closed = False
            semester_known_for_table = True
            continue
        if line.startswith("Course No"):
            if table_closed or not semester_known_for_table:
                semester = ""
                warnings.append(
                    "A course table has no semester heading. Those attempts are kept and marked Requires Review."
                )
                semester_known_for_table = False
            table_open = True
            table_closed = False
            continue
        summary_match = SUMMARY_LINE.search(line)
        if summary_match:
            summaries.append(_summary_row(semester, summary_match))
            table_open = False
            table_closed = True
            continue
        if not table_open or line.startswith("Semester Course Load"):
            continue
        course = _parse_course_line(line)
        if course is None:
            continue
        course["Semester"] = semester or pd.NA
        course["Holdout"] = _is_spring_2026(semester)
        if not semester:
            course["Review"] = "Semester requires review"
        elif course["Holdout"]:
            course["Review"] = f"{HOLD_OUT_LABEL} is excluded from the Phase-1 plan"
        else:
            course["Review"] = ""
        if pd.isna(course["Credit Hours"]):
            warnings.append(f"{course['Course Code']} is missing credit hours on the transcript.")
        attempts.append(course)

    attempt_frame = pd.DataFrame(attempts)
    if attempt_frame.empty:
        warnings.append("No course attempts could be read from the transcript.")
        attempt_frame = pd.DataFrame(
            columns=[
                "Semester", "Course Code", "Course Name", "Credit Hours", "Grade",
                "Grade Point", "Result", "Remarks", "Holdout", "Review",
            ]
        )
    else:
        attempt_frame.insert(0, "Attempt Order", range(1, len(attempt_frame) + 1))
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty and summary_frame["Holdout"].any():
        warnings.append(
            f"{HOLD_OUT_LABEL} appears on this transcript. It is shown for verification and is not used to build the recommendation."
        )
    latest_cgpa = _latest_number(summary_frame, "CGPA")
    return {
        "student": {
            "student_code": student_code,
            "department": department,
            "specialization": specialization,
            "latest_cgpa_on_transcript": latest_cgpa,
            "source_name": "",
        },
        "course_attempts": attempt_frame,
        "semester_summary": summary_frame,
        "warnings": _unique(warnings),
    }


def _parse_course_line(line: str) -> dict[str, object] | None:
    match = COURSE_LINE.match(line)
    if match is None:
        return None
    code, rest = match.group(1), match.group(2)
    remarks = ""
    remark_at = rest.rfind(" R:")
    if remark_at == -1 and rest.startswith("R:"):
        remarks = rest
        rest = ""
    elif remark_at >= 0:
        remarks = rest[remark_at:].strip()
        rest = rest[:remark_at].strip()
    tokens = rest.split()
    grade = ""
    grade_point = pd.NA
    result = ""
    credits = pd.NA
    if len(tokens) >= 2 and tokens[-1] in RESULT_TOKENS:
        result = tokens.pop()
    if tokens and _is_grade_point(tokens):
        grade_point = float(tokens[-1])
        tokens.pop()
    if tokens and tokens[-1] in GRADE_TOKENS:
        grade = tokens.pop()
    if tokens and _is_credit_token(tokens):
        credits = int(tokens[-1])
        tokens.pop()
    if not grade and result == "W":
        grade = "W"
    name = " ".join(tokens).strip()
    return {
        "Course Code": code,
        "Course Name": name,
        "Credit Hours": credits,
        "Grade": grade or pd.NA,
        "Grade Point": grade_point,
        "Result": result or pd.NA,
        "Remarks": remarks or pd.NA,
    }


def _is_grade_point(tokens: list[str]) -> bool:
    token = tokens[-1]
    if len(tokens) < 2 or tokens[-2] not in GRADE_TOKENS:
        return False
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def _is_credit_token(tokens: list[str]) -> bool:
    token = tokens[-1]
    if not re.fullmatch(r"\d+", token):
        return False
    if len(tokens) >= 2 and tokens[-2].casefold() == "level":
        return False
    return int(token) <= 6


def _summary_row(semester: str, match: re.Match[str]) -> dict[str, object]:
    attempted, earned, counted, sgpa, lcgpa, cgpa = match.groups()
    return {
        "Semester": semester or pd.NA,
        "Credits Attempted Total": _optional_int(attempted),
        "Credits Earned": _optional_int(earned),
        "Credits Attempted": _optional_int(counted),
        "SGPA": _optional_float(sgpa),
        "LCGPA": _optional_float(lcgpa),
        "CGPA": _optional_float(cgpa),
        "Holdout": _is_spring_2026(semester),
    }


def _is_spring_2026(semester: str) -> bool:
    text = str(semester).casefold()
    return "2026" in text and "spring" in text


def _latest_number(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def _first_group(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return " ".join(match.group(1).split()) if match else ""


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
