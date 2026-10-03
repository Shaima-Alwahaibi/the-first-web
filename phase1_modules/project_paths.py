"""Repository locations.

This module records folders and source filenames. It does not contain
academic rules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DIR = RAW_DATA_DIR / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DOCS_DIR = PROJECT_ROOT / "docs"
DESCRIPTION_DIR = PROJECT_ROOT / "course_description_sources"

TRANSCRIPT_WORKBOOK = RAW_DATA_DIR / "transcript dataset updated (1).xlsx"
REFERENCE_WORKBOOK = RAW_DATA_DIR / "reference rules - study plan updated (1).xlsx"
PASSING_GRADES_WORKBOOK = RAW_DATA_DIR / "Passing grades.xlsx"

# Reserved for later modeling. Current Tasks 1-10 do not draw random samples.
RANDOM_STATE = 42
HOLDOUT_LABEL = "Spring 2026"
