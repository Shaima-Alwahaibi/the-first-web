"""Load the genuine unresolved Phase-1 limitations from the research report."""

from __future__ import annotations

from pathlib import Path

from app.config import research_root

FALLBACK_LIMITATIONS = (
    "No official General Requirement course list. Study Plan Courses, Elective Pools, Prerequisite Rules, Advising Rules, and Validation Lists were checked. Only the empty slot exists. 273 students keep the status General Requirement Elective — Official Named Pool Required.",
    "No student-level FPMS0001 evidence. Transcript Extracted has no foundation rows. The sample PDFs for STUD-016 and STUD-175 contain other FP codes and do not contain FPMS0001. Five MATH1202 rows stay in manual review.",
    "STUD-232 / MATH1202 and STUD-262 / CSPG1205 have no reconstructable grade and result.",
    "Grade OP is Officially Postponed: not passed, not withdrawn, and not GPA-counted. Is Failed is True on 9 rows and False on 6. The open question is whether those 9 stored fail flags should remain in the difficulty fail count, and whether a latest-attempt OP stays outstanding.",
    "Advising Rules allows mixing at 3 or fewer current-level courses, and a mixing load of 4 courses or 5 when LCGPA is 3.00 or above. Those values are implemented. The phrase satisfies LCGPA/English criteria has no threshold, course, or grade. Mixing Allowed stays Yes for 0 students.",
)


def load_limitations() -> list[str]:
    """Read sections 7 and 8 of the readiness report when the research project is present."""

    root = research_root()
    if root is None:
        return list(FALLBACK_LIMITATIONS)
    report = root / "docs" / "PHASE1_PRE_ML_READINESS_REPORT.md"
    if not report.is_file() and root.name == "phase1_modules":
        report = Path()
    parsed = _parse_report(report) if report.is_file() else []
    return parsed or list(FALLBACK_LIMITATIONS)


def _parse_report(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    collecting = False
    items: list[str] = []
    for line in lines:
        if line.startswith("## 7.") or line.startswith("## 8."):
            collecting = True
            continue
        if collecting and line.startswith("## "):
            break
        if collecting and line.startswith("- "):
            items.append(line[2:].strip())
    return items
