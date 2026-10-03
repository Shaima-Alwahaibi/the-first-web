"""Parser, rule-score, safety, and end-to-end checks for the Phase-1 demo."""

from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd
from pypdf import PdfWriter

WEB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEB_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WEB_ROOT))

from advising_rule_engine import (  # noqa: E402
    CLASS_CURRENT,
    CLASS_ELECTIVE,
    CLASS_FAILED,
    CLASS_FUTURE,
    CLASS_PREVIOUS,
    CLASS_WITHDRAWN,
    STATUS_ALLOWED,
    STATUS_BLOCKED_MIXING,
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_CONDITIONAL,
    STATUS_MANUAL,
)
from app.config import MSG_IMAGE, MSG_SPECIALIZATION, MSG_UNREADABLE, sample_pdfs  # noqa: E402
from app.export_service import recommendation_workbook  # noqa: E402
from app.pipeline_adapter import PipelineDataError, load_student  # noqa: E402
from app.recommendation_service import generate_recommendation, rule_framework  # noqa: E402
from app.transcript_parser import TranscriptReadError, parse_transcript_pdf  # noqa: E402
from rule_based_recommender import SCORE, _score  # noqa: E402


class TranscriptParserTests(unittest.TestCase):
    def test_sample_student_extraction(self) -> None:
        parsed = _parse_sample("STUD-175")
        student = parsed["student"]
        self.assertEqual(student["student_code"], "STUD-175")
        self.assertIn("cyber", str(student["specialization"]).casefold())
        self.assertIn("information technology", str(student["department"]).casefold())
        attempts = parsed["course_attempts"]
        semesters = set(attempts["Semester"].dropna().astype(str))
        self.assertIn("2023 Fall", semesters)
        self.assertIn("2024 Spring", semesters)
        math = attempts.loc[attempts["Course Code"].eq("MATH1202")]
        self.assertGreaterEqual(len(math), 2)
        self.assertEqual(set(math["Grade"].astype(str)), {"W", "C-"})
        self.assertIn("C-", set(math["Grade"].astype(str)))
        graded = attempts.loc[attempts["Grade"].notna()]
        self.assertFalse(graded.empty)
        holdout = attempts.loc[attempts["Holdout"].eq(True), "Course Code"]
        self.assertIn("CSSY3202", set(holdout.astype(str)))

    def test_repeated_routing_attempt_is_preserved(self) -> None:
        attempts = _parse_sample("STUD-175")["course_attempts"]
        routing = attempts.loc[attempts["Course Code"].eq("CSNW2102")]
        self.assertGreaterEqual(len(routing), 2)
        self.assertIn("F", set(routing["Result"].astype(str)))
        self.assertIn("P", set(routing["Result"].astype(str)))

    def test_unreadable_pdf_is_rejected(self) -> None:
        with self.assertRaises(TranscriptReadError) as caught:
            parse_transcript_pdf(b"not a pdf", "bad.pdf")
        self.assertEqual(str(caught.exception), MSG_UNREADABLE)

    def test_blank_pdf_is_rejected_as_image(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buffer = BytesIO()
        writer.write(buffer)
        with self.assertRaises(TranscriptReadError) as caught:
            parse_transcript_pdf(buffer.getvalue(), "blank.pdf")
        self.assertEqual(str(caught.exception), MSG_IMAGE)


class RuleScoreTests(unittest.TestCase):
    def test_engine_scores_match_the_academic_priority_scheme(self) -> None:
        self.assertEqual(_score(CLASS_FAILED, STATUS_ALLOWED), 1.0)
        self.assertEqual(_score(CLASS_WITHDRAWN, STATUS_ALLOWED), 0.90)
        self.assertEqual(_score(CLASS_CURRENT, STATUS_ALLOWED), 0.80)
        self.assertEqual(_score(CLASS_PREVIOUS, STATUS_ALLOWED), 0.80)
        self.assertEqual(_score(CLASS_ELECTIVE, STATUS_ALLOWED), 0.60)
        self.assertEqual(_score(CLASS_FUTURE, STATUS_ALLOWED), 0.50)
        self.assertEqual(_score(CLASS_FAILED, STATUS_BLOCKED_PREREQUISITE), 0.0)
        self.assertEqual(SCORE[CLASS_FAILED], 1.0)
        framework = rule_framework()
        blocked = framework.loc[framework["Academic condition"].eq("Not academically eligible"), "Rule Score"]
        self.assertEqual(float(blocked.iloc[0]), 0.0)


class SafetyTests(unittest.TestCase):
    def test_blocked_and_conditional_courses_are_not_selected(self) -> None:
        result = generate_recommendation(_parse_sample("STUD-175"))
        selected = set(result.selected_courses["Course"].astype(str))
        audit = load_student("STUD-175")["audit"]
        blocked = audit.loc[
            audit["Candidate Status"].isin(
                [STATUS_BLOCKED_PREREQUISITE, STATUS_BLOCKED_MIXING, STATUS_CONDITIONAL, STATUS_MANUAL]
            )
        ]
        self.assertTrue(blocked["Rule-Based Score"].map(float).eq(0.0).all())
        self.assertTrue(selected.isdisjoint(set(blocked["Course Code"].astype(str))))
        self.assertEqual(result.metadata["blocked_courses_selected"], 0)
        self.assertEqual(result.metadata["prerequisite_violations"], 0)
        self.assertFalse(result.selected_courses["Course"].duplicated().any())
        maximum_courses = result.metadata["maximum_courses"]
        maximum_credits = result.metadata["maximum_credits"]
        self.assertLessEqual(len(result.selected_courses), maximum_courses)
        self.assertLessEqual(result.metadata["recommended_credits"], maximum_credits)
        self.assertTrue(result.selected_courses["grade_prediction_score"].isna().all())
        self.assertTrue(result.selected_courses["hybrid_score"].isna().all())

    def test_unknown_student_does_not_invent_a_plan(self) -> None:
        parsed = _parse_sample("STUD-175")
        parsed["student"]["student_code"] = "STUD-999"
        with self.assertRaises(PipelineDataError) as caught:
            generate_recommendation(parsed)
        self.assertEqual(str(caught.exception), MSG_SPECIALIZATION)


class EndToEndTests(unittest.TestCase):
    def test_sample_transcript_reaches_an_exportable_plan(self) -> None:
        parsed = _parse_sample("STUD-175")
        result = generate_recommendation(parsed)
        self.assertEqual(result.metadata["student_code"], "STUD-175")
        self.assertGreater(result.metadata["recommended_courses"], 0)
        self.assertIn("CSSY3202", set(result.selected_courses["Course"].astype(str)))
        self.assertTrue(all(stage["state"] == "confirmed" for stage in result.stages))
        workbook = recommendation_workbook(result)
        sheets = pd.ExcelFile(BytesIO(workbook)).sheet_names
        self.assertEqual(sheets, ["Student Summary", "Recommended Courses", "Blocked Courses"])
        self.assertNotIn(b"C:\\Users", workbook)

    def test_bundled_demo_extract_matches_the_validated_plan(self) -> None:
        from app.pipeline_adapter import _read_directory

        demo = WEB_ROOT / "data" / "demo"
        self.assertFalse((demo / "spring_2026_holdout.xlsx").exists())
        catalog = _read_directory(demo)
        codes = catalog["selected"].loc[
            catalog["selected"]["Student Code"].astype(str).eq("STUD-175"),
            "Recommended Course Code",
        ].astype(str).tolist()
        self.assertEqual(codes, ["CSRM3202", "CSSY3202", "CSSY3203", "CSSE2203"])


def _parse_sample(student_code: str) -> dict[str, object]:
    path = sample_pdfs()[student_code]
    return parse_transcript_pdf(path.read_bytes(), path.name)


if __name__ == "__main__":
    unittest.main()
