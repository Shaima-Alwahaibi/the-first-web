"""Phase-1 academic rule demonstration.

The page collects a transcript, confirms the extraction, and presents the
validated Phase-1 recommendation. It does not calculate rule scores.
"""

from __future__ import annotations

import logging

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
from app.export_service import recommendation_workbook, recommended_csv
from app.pipeline_adapter import PipelineDataError
from app.recommendation_service import generate_recommendation
from app.transcript_parser import ImageTranscriptError, TranscriptReadError, parse_transcript_pdf
from app.transcript_validator import validate_transcript
from app.ui_helpers import (
    inject_styles,
    render_about,
    render_history,
    render_limitations,
    render_methodology,
    render_overview,
    render_profile,
    render_recommendations,
    render_verification,
)

LOGGER = logging.getLogger("phase1_demo")
PAGES = (
    "Overview",
    "Student Transcript",
    "Academic Profile",
    "Course Analysis",
    "Recommendations",
    "Rule Methodology",
    "Limitations",
    "About This Phase",
)


def main() -> None:
    """Run the Streamlit demonstration."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    st.set_page_config(
        page_title="Phase 1 Course Recommendation",
        page_icon="🎓",
        layout="wide",
    )
    inject_styles()
    _initialize()
    page = st.sidebar.radio("Navigation", PAGES, key="nav")
    st.sidebar.caption(MSG_PRIVACY)
    if page == "Overview":
        render_overview()
        _input_panel()
    elif page == "Student Transcript":
        _input_panel()
        _verification_panel()
    elif page == "Academic Profile":
        _require_result(render_profile)
    elif page == "Course Analysis":
        _require_result(render_history)
    elif page == "Recommendations":
        _require_result(_recommendation_page)
    elif page == "Rule Methodology":
        render_methodology()
    elif page == "Limitations":
        render_limitations()
    else:
        render_about()


def _initialize() -> None:
    st.session_state.setdefault("parsed", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("nav", "Overview")
    pending = st.session_state.pop("_pending_page", None)
    if pending:
        st.session_state.nav = pending


def _input_panel() -> None:
    st.subheader("Student transcript")
    samples = sample_pdfs()
    if samples:
        choice = st.selectbox("Sample transcript", list(samples), key="sample_choice")
        if st.button("Run Sample Student", type="primary"):
            if _load_bytes(samples[choice].read_bytes(), samples[choice].name):
                st.session_state._pending_page = "Student Transcript"
                st.rerun()
    else:
        st.warning("Sample transcripts are not available in this copy of the demo.")
    uploaded = st.file_uploader("Upload a UTAS transcript PDF", type=["pdf"])
    if uploaded is not None and st.button("Read uploaded transcript"):
        payload = uploaded.getvalue()
        if not str(uploaded.name).lower().endswith(".pdf"):
            st.error(MSG_TYPE)
        elif not payload:
            st.error(MSG_EMPTY)
        elif len(payload) > MAX_UPLOAD_BYTES:
            st.error(MSG_SIZE)
        elif _load_bytes(payload, uploaded.name):
            st.session_state._pending_page = "Student Transcript"
            st.rerun()


def _load_bytes(payload: bytes, source_name: str) -> bool:
    try:
        parsed = parse_transcript_pdf(payload, source_name)
    except ImageTranscriptError as exc:
        LOGGER.info("Image transcript rejected")
        st.error(str(exc))
        return False
    except TranscriptReadError:
        LOGGER.info("Unreadable transcript rejected")
        st.error(MSG_UNREADABLE)
        return False
    except Exception:
        LOGGER.exception("Transcript parsing failed")
        st.error(MSG_UNREADABLE)
        return False
    st.session_state.parsed = parsed
    st.session_state.result = None
    return True


def _verification_panel() -> None:
    parsed = st.session_state.parsed
    if not parsed:
        st.info("Run a sample student or upload a transcript to extract it.")
        return
    render_verification(parsed)
    validation = validate_transcript(parsed)
    for error in validation["errors"]:
        st.error(error)
    if validation["errors"]:
        return
    if st.button("Generate Recommendation", type="primary"):
        try:
            st.session_state.result = generate_recommendation(parsed)
        except PipelineDataError as exc:
            LOGGER.info("Recommendation stopped: %s", exc.__class__.__name__)
            st.error(str(exc))
            return
        except Exception:
            LOGGER.exception("Recommendation failed")
            st.error("The recommendation could not be completed. Please review the extracted transcript.")
            return
        st.session_state._pending_page = "Academic Profile"
        st.rerun()


def _recommendation_page(result: object) -> None:
    render_recommendations(result)
    workbook = recommendation_workbook(result)
    csv_bytes = recommended_csv(result)
    code = result.metadata.get("student_code") or "student"
    first, second = st.columns(2)
    first.download_button(
        "Download Recommendation Excel",
        data=workbook,
        file_name=f"{code}_phase1_recommendation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    second.download_button(
        "Download recommended courses CSV",
        data=csv_bytes,
        file_name=f"{code}_recommended_courses.csv",
        mime="text/csv",
    )


def _require_result(renderer) -> None:
    result = st.session_state.result
    if result is None:
        st.info("Confirm an extracted transcript and choose Generate Recommendation.")
        if st.button("Go to transcript"):
            st.session_state._pending_page = "Student Transcript"
            st.rerun()
        return
    renderer(result)


if __name__ == "__main__":
    main()
