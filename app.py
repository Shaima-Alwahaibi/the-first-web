"""One-page Phase-1 rule-based recommendation demo.

The page displays results from the validated research modules.
It does not calculate eligibility or rule scores.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import (
    MAX_UPLOAD_BYTES,
    MSG_EMPTY,
    MSG_PRIVACY,
    MSG_SIZE,
    MSG_TYPE,
    MSG_UNREADABLE,
    sample_pdfs,
)
from app.export_service import recommendation_workbook
from app.limitations import manager_limitations
from app.pipeline_adapter import PipelineDataError
from app.recommendation_service import generate_recommendation, rule_framework
from app.transcript_parser import ImageTranscriptError, TranscriptReadError, parse_transcript_pdf
from app.transcript_validator import validate_transcript

LOGGER = logging.getLogger("phase1_demo")


def main() -> None:
    """Show the transcript, the academic summary, and the rule-based plan on one page."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    st.set_page_config(page_title="Phase 1 Course Recommendation", layout="wide")
    st.title("Intelligent Student Course Recommendation System")
    st.subheader("Phase 1 – Rule-Based Recommendation Demo")
    st.write(
        "Upload a student's transcript to analyze academic history, identify remaining courses, "
        "check prerequisites and academic eligibility, rank eligible courses, and generate a valid semester recommendation."
    )
    st.write("This phase uses academic rules only. Machine-learning models will be added in later research phases.")
    st.caption(MSG_PRIVACY)

    _input_section()
    result = st.session_state.get("result")
    if result is None:
        return
    _student_summary(result)
    _history(result)
    _eligible(result)
    _semester_plan(result)
    _blocked(result)
    _rule_table()
    _limitations()
    _download(result)


def _input_section() -> None:
    samples = sample_pdfs()
    left, right = st.columns(2)
    with left:
        st.markdown("**Upload Student Transcript PDF**")
        uploaded = st.file_uploader("Transcript PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded is not None and st.button("Use uploaded transcript"):
            payload = uploaded.getvalue()
            if not str(uploaded.name).lower().endswith(".pdf"):
                st.error(MSG_TYPE)
            elif not payload:
                st.error(MSG_EMPTY)
            elif len(payload) > MAX_UPLOAD_BYTES:
                st.error(MSG_SIZE)
            else:
                _process(payload, uploaded.name)
    with right:
        st.markdown("**Run Sample Student**")
        if not samples:
            st.warning("Sample transcripts are not available in this copy.")
            return
        choice = st.selectbox("Sample", list(samples), label_visibility="collapsed")
        if st.button("Run Sample Student", type="primary"):
            _process(samples[choice].read_bytes(), samples[choice].name)


def _process(payload: bytes, source_name: str) -> None:
    try:
        parsed = parse_transcript_pdf(payload, source_name)
        validation = validate_transcript(parsed)
        if validation["errors"]:
            st.session_state.result = None
            st.error(validation["errors"][0])
            return
        st.session_state.result = generate_recommendation(parsed)
    except ImageTranscriptError as exc:
        LOGGER.info("Image transcript rejected")
        st.session_state.result = None
        st.error(str(exc))
    except TranscriptReadError:
        LOGGER.info("Unreadable transcript rejected")
        st.session_state.result = None
        st.error(MSG_UNREADABLE)
    except PipelineDataError as exc:
        LOGGER.info("Recommendation stopped")
        st.session_state.result = None
        st.error(str(exc))
    except Exception:
        LOGGER.exception("Demo processing failed")
        st.session_state.result = None
        st.error("The transcript could not be processed. Please use a supported UTAS transcript PDF.")


def _student_summary(result) -> None:
    profile = result.student_profile
    st.header("Student Summary")
    st.markdown(
        "\n".join(
            [
                f"**Student Code:** {profile.get('Student Code', 'Requires Review')}",
                f"**Specialization:** {profile.get('Specialization', 'Requires Review')}",
                f"**Current CGPA:** {_number(profile.get('CGPA'))}",
                f"**Latest Semester:** {profile.get('Latest Semester', 'Requires Review')}",
            ]
        )
    )
    holdout = result.metadata.get("holdout_policy")
    if holdout:
        st.caption(str(holdout))
    first, second, third, fourth = st.columns(4)
    first.metric("Completed Courses", _section_count(result, "Completed"))
    second.metric("Failed Courses", _section_count(result, "Failed"))
    third.metric("Withdrawn Courses", _section_count(result, "Withdrawn"))
    fourth.metric("Remaining Courses", _remaining_count(result))
    for warning in result.warnings:
        st.caption(warning)


def _history(result) -> None:
    st.header("Academic History Summary")
    table = _history_table(result)
    if table.empty:
        st.info("No course history was returned for this student.")
        return
    st.dataframe(table, hide_index=True, width="stretch")


def _eligible(result) -> None:
    st.header("All Eligible Courses")
    st.write("Courses that the student is academically allowed to take, ranked by academic priority.")
    frame = result.eligible_courses
    st.markdown(f"**Eligible Courses:** {len(frame)}")
    _show_courses(
        frame,
        {"Course": "Course Code"},
        ["Priority", "Course Code", "Course Name", "Type", "Credits", "Rule Score", "Reason"],
        "No academically eligible courses were returned for this student.",
    )


def _semester_plan(result) -> None:
    st.header("Recommended Semester Plan")
    st.write(
        "Courses selected for the next semester according to academic priority, prerequisites, load limits, and advising rules."
    )
    meta = result.metadata
    st.markdown(
        "\n".join(
            [
                f"**Recommended Courses:** {meta.get('recommended_courses', 0)}",
                f"**Recommended Credits:** {_whole(meta.get('recommended_credits'))}",
                f"**Academic Load:** {meta.get('academic_load', 'Requires Review')}",
            ]
        )
    )
    _show_courses(
        result.selected_courses,
        {"Course": "Course Code", "Credit Hours": "Credits", "Recommendation Reason": "Reason"},
        ["Rank", "Course Code", "Course Name", "Type", "Credits", "Rule Score", "Reason"],
        "The rule engine did not confirm any courses for this student.",
    )
    notes = [
        note for note in result.selected_courses.get("Holdout Note", pd.Series(dtype=str)).tolist()
        if str(note).strip()
    ]
    if notes:
        st.caption(notes[0])


def _show_courses(frame: pd.DataFrame, renames: dict[str, str], columns: list[str], empty_message: str) -> None:
    view = frame.rename(columns=renames)
    show = [column for column in columns if column in view.columns]
    if view.empty:
        st.info(empty_message)
        return
    shown = _scores(view[show], "Rule Score")
    if "Credits" in shown.columns:
        shown["Credits"] = shown["Credits"].map(_whole)
    if "Priority" in shown.columns:
        shown["Priority"] = shown["Priority"].map(_whole)
    st.dataframe(shown, hide_index=True, width="stretch")


def _blocked(result) -> None:
    st.header("Not Eligible / Blocked Courses")
    blocked = result.blocked_courses.rename(columns={"Course": "Course Code", "Score": "Rule Score"})
    show = [column for column in ["Course Code", "Course Name", "Rule Score", "Reason"] if column in blocked.columns]
    if blocked.empty:
        st.success("No blocked courses were returned for this student.")
        return
    st.dataframe(_scores(blocked[show], "Rule Score"), hide_index=True, width="stretch")


def _rule_table() -> None:
    st.header("How Rule Score Works")
    st.write("Eligibility is decided before a rule score is used. A course that is not eligible scores 0.00 and is not recommended.")
    table = rule_framework().rename(columns={"Academic condition": "Rule Condition", "Advisor meaning": "Meaning"})
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Rule Score": st.column_config.NumberColumn(format="%.2f")},
    )


def _limitations() -> None:
    st.header("Current Limitations")
    for item in manager_limitations():
        st.write(f"- {item}")


def _download(result) -> None:
    code = result.metadata.get("student_code") or "student"
    st.download_button(
        "Download Recommendation Excel",
        data=recommendation_workbook(result),
        file_name=f"{code}_phase1_recommendation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _history_table(result) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    repeat_frame = result.course_history.get("Repeated", pd.DataFrame())
    repeated_codes = set()
    if isinstance(repeat_frame, pd.DataFrame) and not repeat_frame.empty and "Course Code" in repeat_frame.columns:
        repeated_codes = set(repeat_frame["Course Code"].astype(str))
    for status in ("Completed", "Failed", "Withdrawn"):
        frame = result.course_history.get(status, pd.DataFrame())
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for _, row in frame.iterrows():
            code = _text(row.get("Course Code"))
            name = _text(row.get("Course Name") or row.get("Course Title"))
            label = "Repeated" if code in repeated_codes else status
            rows.append({"Course Code": code, "Course Name": name, "Grade": _first_grade(row), "Status": label})
            seen.add(code)
    repeat_frame = result.course_history.get("Repeated", pd.DataFrame())
    if isinstance(repeat_frame, pd.DataFrame) and not repeat_frame.empty:
        for _, row in repeat_frame.iterrows():
            code = _text(row.get("Course Code"))
            if not code or code in seen:
                continue
            name = _text(row.get("Course Name") or row.get("Course Title"))
            grade = _text(row.get("Grade")) or _first_grade(row)
            rows.append({"Course Code": code, "Course Name": name, "Grade": grade, "Status": "Repeated"})
            seen.add(code)
    return pd.DataFrame(rows, columns=["Course Code", "Course Name", "Grade", "Status"])


def _first_grade(row) -> str:
    for column in ("Completion Grade", "Latest Grade", "Grade"):
        if column in row.index:
            text = _text(row.get(column))
            if text:
                return text
    return ""


def _section_count(result, name: str) -> int:
    frame = result.course_history.get(name)
    if not isinstance(frame, pd.DataFrame):
        return 0
    return int(len(frame))


def _remaining_count(result) -> int:
    value = result.student_profile.get("Remaining Courses")
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number):
        return int(number)
    frame = result.course_history.get("Remaining")
    if isinstance(frame, pd.DataFrame):
        return int(len(frame))
    return 0


def _scores(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    view = frame.copy()
    if column in view.columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return view


def _number(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "Requires Review"
    return f"{float(number):.2f}"


def _whole(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "Requires Review"
    return str(int(round(float(number))))


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text in {"nan", "None"} else text


if __name__ == "__main__":
    main()
