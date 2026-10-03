"""Presentation helpers. Academic decisions stay in the research modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import HOLD_OUT_LABEL, MSG_PRIVACY
from app.limitations import load_limitations
from app.recommendation_service import RecommendationResult, rule_framework

PHASE_DONE = (
    "Transcript processing",
    "Academic profile",
    "Study-plan matching",
    "Remaining-course analysis",
    "Prerequisite checking",
    "Academic eligibility",
    "Rule-based recommendations",
)
PHASE_LATER = (
    "Grade prediction",
    "Collaborative filtering",
    "Content-based similarity",
    "Transformer models",
    "Hybrid recommendation",
    "CGPA impact prediction",
)
HISTORY_COLUMNS = {
    "Completed": [
        "Course Code", "Course Name", "Completion Semester", "Completion Grade",
        "Final Status", "Attempt Count",
    ],
    "Failed": ["Course Code", "Course Name", "Latest Attempt Semester", "Latest Grade", "Final Status", "Attempt Count"],
    "Withdrawn": ["Course Code", "Course Name", "Latest Attempt Semester", "Latest Grade", "Final Status", "Attempt Count"],
    "Repeated": [
        "Course Code", "Semester", "Attempt Number", "Grade", "Result",
        "Derived Repeat Type", "Previous Attempt Outcome", "Repeat Resolution Status",
    ],
    "Remaining": [
        "Course Code", "Course Title", "Course Type", "Study Plan Level", "Remaining Status",
        "Remaining Reason", "Credit Hours",
    ],
}


def inject_styles() -> None:
    """Apply a quiet academic theme."""

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; max-width: 1180px;}
        h1 {font-size: 2rem; letter-spacing: -0.02em;}
        .phase-card {
            border: 1px solid #D5DEE8;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: #F7FAFC;
            margin-bottom: 0.8rem;
        }
        .notice {
            border-left: 4px solid #1F4E79;
            padding: 0.6rem 0.8rem;
            background: #F3F7FB;
            margin: 0.6rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    """Show the Phase-1 purpose and what is intentionally not implemented."""

    st.title("Intelligent Student Course Recommendation System")
    st.subheader("Phase 1 – Academic Rule-Based Recommendation Prototype")
    st.write(
        "Upload a student's academic transcript to demonstrate how the system analyzes "
        "academic history, applies study-plan and prerequisite rules, identifies remaining "
        "requirements, and recommends eligible courses using academic priority rules."
    )
    st.markdown('<div class="notice">' + MSG_PRIVACY + "</div>", unsafe_allow_html=True)
    done = "<br>".join(f"✓ {item}" for item in PHASE_DONE)
    later = "<br>".join(f"○ {item}" for item in PHASE_LATER)
    left, right = st.columns(2)
    left.markdown(f'<div class="phase-card"><strong>Completed in this prototype</strong><br>{done}</div>', unsafe_allow_html=True)
    right.markdown(f'<div class="phase-card"><strong>Future research</strong><br>{later}</div>', unsafe_allow_html=True)
    st.info(
        "This demonstration stops at the completed rule-based recommender. "
        "It does not train a model and it does not score courses with collaborative filtering, "
        "content similarity, transformers, grade prediction, or CGPA prediction."
    )


def render_verification(parsed: dict[str, object]) -> None:
    """Show the extracted transcript before a recommendation is requested."""

    st.header("Verify Extracted Transcript")
    student = parsed.get("student") or {}
    attempts = parsed.get("course_attempts")
    count = len(attempts) if isinstance(attempts, pd.DataFrame) else 0
    cgpa = student.get("latest_cgpa_on_transcript")
    cgpa_text = f"{cgpa:.2f}" if isinstance(cgpa, float) else "Not printed"
    st.markdown(
        "\n".join(
            [
                f"**Student Code:** {student.get('student_code') or 'Not found'}",
                f"**Department:** {student.get('department') or 'Not found'}",
                f"**Specialization printed on the transcript:** {student.get('specialization') or 'Not found'}",
                f"**Detected Course Attempts:** {count}",
                f"**Latest CGPA printed on the transcript:** {cgpa_text}",
            ]
        )
    )
    st.caption(
        f"Raw extraction is shown here. Academic status and the recommended plan are produced "
        f"only after confirmation, from the validated Phase-1 pipeline. {HOLD_OUT_LABEL} rows stay visible "
        "and are not used to complete a course."
    )
    warnings = parsed.get("warnings") or []
    if warnings:
        st.warning("Extraction warnings")
        for warning in warnings:
            st.write(f"⚠ {warning}")
    with st.expander("View Extracted Courses", expanded=True):
        if isinstance(attempts, pd.DataFrame) and not attempts.empty:
            view = attempts.copy()
            view["Pipeline use"] = view["Holdout"].map(
                lambda holdout: f"Excluded — {HOLD_OUT_LABEL}" if holdout else "Pre-holdout transcript row"
            )
            columns = [
                "Semester", "Course Code", "Course Name", "Credit Hours", "Grade", "Result", "Pipeline use",
            ]
            st.dataframe(view[[column for column in columns if column in view.columns]], hide_index=True, width="stretch")
        else:
            st.error("No course attempts were extracted.")


def render_stages(result: RecommendationResult) -> None:
    """Show a stage only as complete when that stage actually succeeded."""

    st.subheader("Processing status")
    for stage in result.stages:
        state = stage["state"]
        if state == "confirmed":
            st.write(f"✓ {stage['label']}")
        elif state == "review":
            st.write(f"⚠ Requires Review — {stage['label']}")
        else:
            st.write(f"❌ {stage['label']}")


def render_profile(result: RecommendationResult) -> None:
    """Show profile values that the validated pipeline actually produced."""

    st.header("Student Academic Profile")
    render_stages(result)
    cards = result.student_profile
    labels = list(cards)
    if not labels:
        st.warning("The validated profile did not contain displayable academic fields.")
        return
    columns = st.columns(3)
    for index, label in enumerate(labels):
        value = cards[label]
        if isinstance(value, float):
            value = f"{value:.2f}"
        columns[index % 3].metric(label, value)


def render_history(result: RecommendationResult) -> None:
    """Show pipeline history sections, including every stored repeat attempt."""

    st.header("Academic History")
    st.caption("Completed, failed, withdrawn, repeated, and remaining rows come from the validated Phase-1 outputs.")
    tabs = st.tabs(list(HISTORY_COLUMNS))
    for tab, (name, columns) in zip(tabs, HISTORY_COLUMNS.items()):
        frame = result.course_history.get(name, pd.DataFrame())
        with tab:
            _show_frame(frame, columns)


def render_recommendations(result: RecommendationResult) -> None:
    """Show the confirmed plan, blocked courses, and traceable engine fields."""

    meta = result.metadata
    st.header("Recommended Study Plan")
    left, right = st.columns(2)
    left.markdown(
        "\n".join(
            [
                f"**Recommended Courses:** {meta.get('recommended_courses')}",
                f"**Recommended Credits:** {_credits(meta.get('recommended_credits'))}",
                f"**Academic Load:** {meta.get('academic_load')}",
            ]
        )
    )
    right.markdown(
        "\n".join(
            [
                f"**Prerequisite Violations:** {meta.get('prerequisite_violations')}",
                f"**Blocked Courses Selected:** {meta.get('blocked_courses_selected')}",
                f"**Study Plan:** {meta.get('assigned_pathway') or 'Requires Review'}",
            ]
        )
    )
    view = result.selected_courses[
        [column for column in [
            "Rank", "Course", "Course Name", "Type", "Rule Score", "Status", "Recommendation Reason",
        ] if column in result.selected_courses.columns]
    ]
    st.dataframe(_format_scores(view), hide_index=True, width="stretch")
    notes = result.selected_courses["Holdout Note"] if "Holdout Note" in result.selected_courses.columns else pd.Series(dtype=str)
    visible_notes = [note for note in notes.tolist() if str(note).strip()]
    if visible_notes:
        st.info("Holdout handling for courses that also appear on the uploaded transcript")
        for note in dict.fromkeys(visible_notes):
            st.write(note)

    st.header("Blocked or Not Currently Eligible")
    st.caption("These courses keep the score recorded by the rule engine. A blocked course is not added to the confirmed plan.")
    blocked = result.blocked_courses.copy()
    if blocked.empty:
        st.success("✅ Confirmed — no blocked or unconfirmed candidate was returned for this student.")
    else:
        statuses = ["All blocked or unconfirmed"] + sorted(blocked["Status"].dropna().astype(str).unique())
        chosen = st.selectbox("Blocked status", statuses)
        if chosen != "All blocked or unconfirmed":
            blocked = blocked.loc[blocked["Status"].astype(str).eq(chosen)]
        view = blocked[[column for column in ["Course", "Score", "Status", "Reason"] if column in blocked.columns]]
        st.dataframe(_format_scores(view, "Score"), hide_index=True, width="stretch")

    with st.expander("View Recommendation Details"):
        detail_columns = [
            "Rank", "Course", "Priority Class", "Rule Score", "Status", "Study Plan",
            "Remaining Requirement", "Recommendation Reason", "Holdout Note",
        ]
        _show_frame(result.selected_courses, detail_columns)
        st.markdown("**Prerequisite evidence for the confirmed courses**")
        if result.prerequisite_results.empty or result.selected_courses.empty:
            st.write("⚠ Requires Review — prerequisite rows were not available for the confirmed courses.")
        else:
            codes = set(result.selected_courses["Course"].astype(str))
            evidence = result.prerequisite_results.loc[
                result.prerequisite_results["Course Code"].astype(str).isin(codes)
            ]
            _show_frame(
                evidence,
                [
                    "Course Code", "Eligibility Status", "Eligibility Reason",
                    "Satisfied Prerequisites", "Missing Prerequisites", "Candidate Scope",
                    "Prerequisite Rule Normalized",
                ],
            )

    if result.warnings:
        st.subheader("Warnings")
        for warning in result.warnings:
            st.write(f"⚠ {warning}")
    render_manager_summary(result)


def render_manager_summary(result: RecommendationResult) -> None:
    """Close the result with the Phase-1 scope statement."""

    st.header("Phase-1 Result")
    code = result.metadata.get("student_code") or "the student"
    st.write(
        f"The system processed {code}'s academic history, matched the relevant study plan, "
        "identified outstanding course requirements, checked academic prerequisites and eligibility, "
        "and generated a rule-based recommended study plan."
    )
    st.write("No machine-learning model is used at this stage.")
    st.write(
        "The output forms the deterministic academic foundation for the next research phase: "
        "grade prediction and hybrid recommendation."
    )
    st.caption(str(result.metadata.get("holdout_policy", "")))


def render_methodology() -> None:
    """Explain the completed deterministic sequence."""

    st.header("How does the Phase-1 recommender work?")
    framework = rule_framework()
    display = framework.copy()
    display["Rule Score"] = display["Rule Score"].map(lambda value: f"{float(value):.2f}")
    st.dataframe(display, hide_index=True, width="stretch")
    st.write(
        "Rule scores represent academic priority among eligible candidates. Eligibility constraints "
        "such as prerequisites, study-plan requirements, pathway restrictions, and level-mixing rules "
        "are applied before a course is confirmed for recommendation."
    )
    st.subheader("Methodology")
    phases = (
        ("Phase 1 — Input", "Student transcript PDF."),
        ("Phase 2 — Extraction", "Convert transcript information into structured academic records."),
        ("Phase 3 — Validation", "Verify student information, courses, grades, semesters, and extraction quality."),
        ("Phase 4 — Academic Profile", "Determine academic history and current progression."),
        ("Phase 5 — Curriculum Mapping", "Identify the student's correct study plan/specialization."),
        ("Phase 6 — Remaining Requirements", "Determine completed and outstanding requirements."),
        ("Phase 7 — Academic Eligibility", "Check passing rules, prerequisites, pathway conditions, level restrictions, and advising policies."),
        ("Phase 8 — Rule-Based Prioritization", "Apply normalized academic-priority scores."),
        ("Phase 9 — Study Plan Generation", "Select academically valid recommendations within applicable workload rules."),
        ("Phase 10 — Explainability", "Show why each course is recommended, blocked, or requires review."),
    )
    for title, body in phases:
        st.markdown(f"**{title}**")
        st.write(body)
    st.success("This completes the deterministic academic foundation required before introducing machine-learning ranking methods.")


def render_limitations() -> None:
    """Show unresolved items recorded by the research project."""

    st.header("Current Phase Limitations")
    st.write("These items are unresolved in the Phase-1 research record. The demo does not invent a rule to remove them.")
    for item in load_limitations():
        st.write(f"⚠ {item}")


def render_about() -> None:
    """State the research boundary of this web demonstration."""

    st.header("About This Phase")
    st.write(
        "This is the Phase-1 academic-rule recommendation prototype built from the completed "
        "data preparation, academic-profile, curriculum, prerequisite, eligibility, and "
        "rule-based recommendation work."
    )
    st.markdown(
        """
Transcript PDF  
↓  
Transcript Extraction  
↓  
Validation  
↓  
Academic Profile  
↓  
Study Plan  
↓  
Remaining Courses  
↓  
Prerequisite Checking  
↓  
Academic Eligibility  
↓  
Rule-Based Priority  
↓  
Recommended Study Plan
        """
    )
    st.write(f"{HOLD_OUT_LABEL} remains an evaluation holdout. This application does not train on it and does not merge it into the historical profile.")
    st.caption(MSG_PRIVACY)


def _show_frame(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame is None or frame.empty:
        st.info("None recorded for this student.")
        return
    keep = [column for column in columns if column in frame.columns]
    st.dataframe(frame[keep] if keep else frame, hide_index=True, width="stretch")


def _format_scores(frame: pd.DataFrame, column: str = "Rule Score") -> pd.DataFrame:
    view = frame.copy()
    if column in view.columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return view


def _credits(value: object) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "Requires Review"
