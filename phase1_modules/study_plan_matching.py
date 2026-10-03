"""Map historical students to official study-plan pathways.

The mapper reads the historical student profile and the official study-plan
courses. A second pass can resolve a manual-review student only from Fall 2025
history and the official plan. It does not calculate remaining courses.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from preparation_checks import normalize_course_code, write_atomic

PATH_COMMON = "Common Year 1"
PATH_SE = "Software Engineering"
PATH_NCS = "Network Computing and Security"
PATH_DSAI = "Data Science and Artificial Intelligence"
PATH_CYBER = "Cyber and Information Security"
PATH_REVIEW = "Manual Review"

SOURCE_SE_YEAR2 = "Software Engineering Year 2"
SOURCE_NCS_YEAR2 = "Network Computing and Security Year 2"

METHOD_EXPLICIT = "Explicit Historical Specialization"
METHOD_UNIQUE = "Unique Pathway Course Evidence"
METHOD_PATTERN = "Study Plan Course Pattern"
METHOD_COMMON = "Common Year"
METHOD_AMBIGUOUS = "Ambiguous"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_REVIEW = "Review"

LEVEL_DIPLOMA = "Diploma"
LEVEL_ADVANCED = "Advanced Diploma"
LEVEL_BACHELOR = "Bachelor"
UNKNOWN_LEVEL = "Unknown/Manual Review"
MISSING_SPECIALIZATION = "Not available before Spring 2026"
DEFERRED_REMAINING = "Deferred to Remaining Course Task"
HOLDOUT_TOKEN = "2026"

ASSIGNED_PATHWAYS = (PATH_COMMON, PATH_SE, PATH_NCS, PATH_DSAI, PATH_CYBER, PATH_REVIEW)
EXPLICIT_SPECIALIZATIONS = {
    PATH_SE.casefold(): PATH_SE,
    PATH_NCS.casefold(): PATH_NCS,
    PATH_DSAI.casefold(): PATH_DSAI,
    PATH_CYBER.casefold(): PATH_CYBER,
}
PLAN_LEVELS = {LEVEL_DIPLOMA.casefold(): LEVEL_DIPLOMA, LEVEL_ADVANCED.casefold(): LEVEL_ADVANCED, LEVEL_BACHELOR.casefold(): LEVEL_BACHELOR}
HIGHER_PATHWAYS = (PATH_SE, PATH_DSAI, PATH_CYBER)

REQUIRED_REFERENCE_SHEETS = (
    "Study Plan Courses",
    "Prerequisite Rules",
    "Elective Pools",
    "Advising Rules",
    "Validation Lists",
)
STUDY_PLAN_COLUMNS = (
    "Specialization Path",
    "Level",
    "Year",
    "Course Code",
    "Course Title",
    "Course Type",
    "Requirement Type",
    "Is Core",
    "Is Elective",
    "Elective Pool",
)
PROFILE_COLUMNS = (
    "Student Code",
    "Current Level",
    "Current Specialization",
    "Latest Semester",
    "Completed Courses",
    "Failed Courses",
    "Withdrawn Courses",
    "Repeated Courses",
)

MAPPING_COLUMNS = [
    "Student Code",
    "Current Level",
    "Profile Specialization",
    "Assigned Pathway",
    "Pathway Stage",
    "Mapping Method",
    "Mapping Evidence",
    "Mapping Confidence",
    "Manual Review Required",
    "Completed Courses Count",
    "Matched Pathway Courses Count",
    "Completed Courses",
    "Remaining Core Courses",
    "Completed Electives",
    "Pending Electives",
]
EVIDENCE_COLUMNS = [
    "Student Code",
    "Course Code",
    "Course Status",
    "Matched Study Plan Pathway",
    "Study Plan Year",
    "Study Plan Level",
    "Evidence Type",
]


def load_student_profiles(path: Path) -> pd.DataFrame:
    """Read the historical student-profile sheet."""
    profiles = pd.read_excel(path, sheet_name="Student Academic Profiles")
    validate_required_columns(profiles, PROFILE_COLUMNS, "student profiles")
    return profiles


def load_study_plan(path: Path) -> dict[str, pd.DataFrame]:
    """Read the official reference workbook, including sheets used by later tasks."""
    workbook = pd.ExcelFile(path)
    missing = [name for name in REQUIRED_REFERENCE_SHEETS if name not in workbook.sheet_names]
    if missing:
        raise RuntimeError("Study-plan workbook is missing sheets: " + ", ".join(missing))
    return {name: pd.read_excel(workbook, sheet_name=name) for name in REQUIRED_REFERENCE_SHEETS}


def describe_reference_sheets(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return row counts, column counts, and column names for each required sheet."""
    rows = []
    for name in REQUIRED_REFERENCE_SHEETS:
        frame = sheets[name]
        rows.append({
            "Sheet": name,
            "Rows": int(len(frame)),
            "Columns": int(len(frame.columns)),
            "Column Names": ", ".join(str(column) for column in frame.columns),
        })
    return pd.DataFrame(rows)


def validate_required_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], label: str) -> None:
    """Raise when a required column is absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: " + ", ".join(missing))


def validate_student_profiles(profiles: pd.DataFrame) -> dict[str, int]:
    """Check the historical profile boundary before any pathway is assigned."""
    validate_required_columns(profiles, PROFILE_COLUMNS, "student profiles")
    if not profiles["Student Code"].is_unique:
        raise RuntimeError("Student profiles do not have unique Student Codes.")
    leakage = int(profiles["Latest Semester"].astype(str).str.contains(HOLDOUT_TOKEN).sum())
    if leakage != 0:
        raise RuntimeError(f"Spring 2026 leakage in student profiles is {leakage}.")
    return {
        "rows": int(len(profiles)),
        "unique_students": int(profiles["Student Code"].nunique()),
        "spring_2026_leakage": leakage,
    }


def validate_study_plan_courses(plan: pd.DataFrame) -> None:
    """Check the study-plan course sheet before pathway sets are built."""
    validate_required_columns(plan, STUDY_PLAN_COLUMNS, "Study Plan Courses")
    if plan.empty:
        raise RuntimeError("Study Plan Courses has no rows.")


def normalize_text(value: object) -> str:
    """Trim and collapse whitespace without changing token meaning."""
    if _missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_study_plan_courses(plan: pd.DataFrame) -> pd.DataFrame:
    """Normalize study-plan labels and keep the source text beside each field."""
    validate_study_plan_courses(plan)
    work = plan.copy()
    work["Course Code Original"] = work["Course Code"]
    work["Course Code"] = work["Course Code"].map(normalize_course_code)
    for column in ("Course Title", "Year", "Level", "Specialization Path", "Course Type", "Requirement Type", "Elective Pool"):
        work[f"{column} Original"] = work[column]
        work[column] = work[column].map(_normalize_label)
    return work


def normalization_audit(plan: pd.DataFrame) -> pd.DataFrame:
    """Count study-plan values whose display text changes during normalization."""
    normalized = normalize_study_plan_courses(plan)
    rows = []
    for column in ("Course Code", "Course Title", "Year", "Level", "Specialization Path", "Course Type", "Requirement Type", "Elective Pool"):
        changed = sum(
            _display_token(original) != _display_token(current)
            for original, current in zip(normalized[f"{column} Original"], normalized[column])
        )
        rows.append({"Field": column, "Rows": int(len(normalized)), "Changed Values": int(changed)})
    return pd.DataFrame(rows)


def build_pathway_catalog(plan: pd.DataFrame) -> dict[str, object]:
    """Build pathway course sets from official study-plan rows.

    Elective placeholders are not pathway evidence. Shared courses are removed
    from the sets used to distinguish one pathway from another.
    """
    work = normalize_study_plan_courses(plan)
    work["Is Core"] = work["Is Core"].map(_yes)
    cores = work.loc[work["Is Core"] & work["Course Code"].notna()].copy()
    common = _core_codes(cores, PATH_COMMON, LEVEL_DIPLOMA)
    se_year2 = _core_codes(cores, SOURCE_SE_YEAR2, LEVEL_DIPLOMA)
    ncs_year2 = _core_codes(cores, SOURCE_NCS_YEAR2, LEVEL_DIPLOMA)
    blocks = {
        PATH_SE: _core_codes(cores, PATH_SE),
        PATH_DSAI: _core_codes(cores, PATH_DSAI),
        PATH_CYBER: _core_codes(cores, PATH_CYBER),
    }
    occupied = common | se_year2 | ncs_year2
    signatures = {}
    for pathway, codes in blocks.items():
        others = set().union(*(blocks[name] for name in blocks if name != pathway))
        signatures[pathway] = codes - others - occupied
    placements: dict[str, list[dict[str, str]]] = {}
    for record in cores.to_dict(orient="records"):
        placements.setdefault(str(record["Course Code"]), []).append({
            "Pathway": record["Specialization Path"],
            "Year": record["Year"],
            "Level": record["Level"],
        })
    return {
        "common": common,
        "se_year2": se_year2,
        "ncs_year2": ncs_year2,
        "se_year2_unique": se_year2 - ncs_year2 - common,
        "ncs_year2_unique": ncs_year2 - se_year2 - common,
        "blocks": blocks,
        "signatures": signatures,
        "placements": placements,
        "all_cores": set(cores["Course Code"].astype(str)),
    }


def build_study_plan_mapping(
    profiles: pd.DataFrame,
    plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign one pathway row per student and return mapping, evidence, and checks."""
    counts = validate_student_profiles(profiles)
    catalog = build_pathway_catalog(plan)
    mapping_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for profile in profiles.to_dict(orient="records"):
        completed = parse_course_list(profile.get("Completed Courses"))
        failed = parse_course_list(profile.get("Failed Courses"))
        withdrawn = parse_course_list(profile.get("Withdrawn Courses"))
        repeated = parse_course_list(profile.get("Repeated Courses"))
        decision = infer_student_pathway(
            level=profile.get("Current Level"),
            specialization=profile.get("Current Specialization"),
            completed=completed,
            failed=failed,
            withdrawn=withdrawn,
            catalog=catalog,
        )
        mapping_rows.append({
            "Student Code": str(profile["Student Code"]),
            "Current Level": normalize_text(profile.get("Current Level")) or UNKNOWN_LEVEL,
            "Profile Specialization": normalize_text(profile.get("Current Specialization")) or MISSING_SPECIALIZATION,
            "Assigned Pathway": decision["pathway"],
            "Pathway Stage": decision["stage"],
            "Mapping Method": decision["method"],
            "Mapping Evidence": decision["evidence"],
            "Mapping Confidence": decision["confidence"],
            "Manual Review Required": "Yes" if decision["review"] else "No",
            "Completed Courses Count": len(completed),
            "Matched Pathway Courses Count": decision["matched_count"],
            "Completed Courses": _display_course_list(profile.get("Completed Courses")),
            "Remaining Core Courses": DEFERRED_REMAINING,
            "Completed Electives": DEFERRED_REMAINING,
            "Pending Electives": DEFERRED_REMAINING,
        })
        evidence_rows.extend(_evidence_rows(
            student_code=str(profile["Student Code"]),
            decision=decision,
            completed=completed,
            failed=failed,
            withdrawn=withdrawn,
            repeated=repeated,
            catalog=catalog,
        ))
    mapping = pd.DataFrame(mapping_rows, columns=MAPPING_COLUMNS)
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)
    mapping = mapping.sort_values("Student Code", kind="mergesort").reset_index(drop=True)
    if not evidence.empty:
        evidence = evidence.sort_values(["Student Code", "Course Code"], kind="mergesort").reset_index(drop=True)
    validation = validate_pathway_assignment(mapping, profiles, spring_2026_leakage=counts["spring_2026_leakage"])
    return mapping, evidence, validation


def infer_student_pathway(
    *,
    level: object,
    specialization: object,
    completed: set[str],
    failed: set[str],
    withdrawn: set[str],
    catalog: dict[str, object],
) -> dict[str, object]:
    """Choose one official pathway, or manual review when the evidence is not unique."""
    canonical_level = _canonical_level(level)
    explicit = _explicit_specialization(specialization)
    supporting = failed | withdrawn
    year2 = _select_track(
        ((PATH_SE, catalog["se_year2_unique"]), (PATH_NCS, catalog["ncs_year2_unique"])),
        completed,
        supporting,
    )
    higher = _select_track(
        tuple((name, catalog["signatures"][name]) for name in HIGHER_PATHWAYS),
        completed,
        supporting,
    )
    historical = completed | failed | withdrawn
    if year2["status"] == "conflict" or higher["status"] == "conflict":
        return _review("Conflicting pathway-specific courses", _conflict_evidence(year2, higher), catalog, completed)
    if _tracks_conflict(year2, higher):
        return _review("Conflicting pathway-specific courses", _conflict_evidence(year2, higher), catalog, completed)
    if explicit is not None:
        contradiction = _explicit_contradiction(explicit, year2, higher)
        if contradiction is not None:
            return _review("Conflicting pathway-specific courses", contradiction, catalog, completed)
        return _assign_explicit(explicit, canonical_level, year2, higher, catalog, completed)
    if year2["status"] == "selected" and year2["label"] == PATH_SE:
        return _assign_software_track(canonical_level, year2, higher, catalog, completed)
    if year2["status"] == "selected" and year2["label"] == PATH_NCS:
        return _assign_network_track(canonical_level, year2, higher, catalog, completed)
    if higher["status"] == "selected":
        return _assignment(
            pathway=higher["label"],
            stage=_stage_for_pathway(higher["label"], canonical_level, higher["codes"], catalog),
            method=METHOD_UNIQUE if higher["strength"] == "completed" else METHOD_PATTERN,
            evidence=_course_evidence("Pathway-specific courses", higher["codes"]),
            confidence=CONFIDENCE_HIGH if higher["strength"] == "completed" else CONFIDENCE_MEDIUM,
            review=False,
            matched_count=_matched_count(higher["label"], canonical_level, completed, catalog),
            evidence_codes=higher["codes"],
        )
    if _within_common_year(historical, catalog):
        common_codes = tuple(sorted(historical & catalog["common"]))
        return _assignment(
            pathway=PATH_COMMON,
            stage="Year 1",
            method=METHOD_COMMON,
            evidence="Historical courses stay within Common Year 1" + (_suffix(common_codes)),
            confidence=CONFIDENCE_HIGH if common_codes else CONFIDENCE_MEDIUM,
            review=False,
            matched_count=len(completed & catalog["common"]),
            evidence_codes=common_codes,
        )
    return _review("No pathway-specific historical course evidence", "No completed, failed, or withdrawn course identifies one official pathway", catalog, completed)


def validate_pathway_assignment(
    mapping: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    spring_2026_leakage: int,
) -> pd.DataFrame:
    """Raise on broken mapping integrity and return the check table."""
    if not mapping["Student Code"].is_unique:
        raise RuntimeError('mapping["Student Code"].is_unique failed.')
    if len(mapping) != len(profiles):
        raise RuntimeError("Mapping row count does not equal the student-profile count.")
    if set(mapping["Student Code"]) != set(profiles["Student Code"].astype(str)):
        raise RuntimeError("Mapping students do not match the student profiles.")
    unknown = sorted(set(mapping["Assigned Pathway"]) - set(ASSIGNED_PATHWAYS))
    if unknown:
        raise RuntimeError("Assigned pathways are outside the official study plan: " + ", ".join(unknown))
    automatic = mapping.loc[mapping["Manual Review Required"].eq("No")]
    if automatic["Mapping Evidence"].map(_missing).any():
        raise RuntimeError("An automatic pathway assignment has no evidence.")
    review = mapping.loc[mapping["Manual Review Required"].eq("Yes")]
    if not review.empty and review["Mapping Evidence"].map(_missing).any():
        raise RuntimeError("A manual-review pathway has no reason.")
    if review["Assigned Pathway"].ne(PATH_REVIEW).any():
        raise RuntimeError("A manual-review row has a pathway other than Manual Review.")
    if (mapping["Assigned Pathway"].eq(PATH_REVIEW) & mapping["Manual Review Required"].eq("No")).any():
        raise RuntimeError("Manual Review was assigned without the review flag.")
    counts = mapping["Assigned Pathway"].value_counts()
    conflicting = int(mapping["Mapping Evidence"].astype(str).str.startswith("Conflicting pathway-specific courses").sum())
    checks = [
        _check("Total students", len(mapping), fail=len(mapping) != len(profiles)),
        _check("Unique students", mapping["Student Code"].nunique(), fail=not mapping["Student Code"].is_unique),
        _check("Mapped automatically", int(mapping["Manual Review Required"].eq("No").sum())),
        _check("Common Year 1", int(counts.get(PATH_COMMON, 0))),
        _check("Software Engineering", int(counts.get(PATH_SE, 0))),
        _check("Network Computing and Security", int(counts.get(PATH_NCS, 0))),
        _check("Data Science and Artificial Intelligence", int(counts.get(PATH_DSAI, 0))),
        _check("Cyber and Information Security", int(counts.get(PATH_CYBER, 0))),
        _check("Manual Review", int(counts.get(PATH_REVIEW, 0)), review=int(counts.get(PATH_REVIEW, 0)) > 0),
        _check("Missing level", int(mapping["Current Level"].eq(UNKNOWN_LEVEL).sum()), review=bool(mapping["Current Level"].eq(UNKNOWN_LEVEL).any())),
        _check("Missing profile specialization", int(mapping["Profile Specialization"].eq(MISSING_SPECIALIZATION).sum())),
        _check("Students with conflicting pathway evidence", conflicting, review=conflicting > 0),
        _check("Spring 2026 leakage", spring_2026_leakage, fail=spring_2026_leakage != 0),
        _check("Duplicate Student Codes", int(mapping["Student Code"].duplicated().sum()), fail=bool(mapping["Student Code"].duplicated().any())),
    ]
    return pd.DataFrame(checks, columns=["Check", "Result", "Status"])


def parse_course_list(value: object) -> set[str]:
    """Split a profile course list into normalized codes. Python literals are read safely."""
    if _missing(value):
        return set()
    text = str(value).strip()
    if text == "" or text.casefold() == "nan":
        return set()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"Course list could not be parsed: {text[:80]}") from exc
        items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    else:
        items = re.split(r"[;|,]", text)
    codes = set()
    for item in items:
        code = normalize_course_code(item)
        if not _missing(code):
            codes.add(str(code))
    return codes


def pathway_catalog_summary(catalog: dict[str, object]) -> pd.DataFrame:
    """Return the pathway sets used for matching, with one row per set."""
    rows = [
        ("Common Year 1", PATH_COMMON, len(catalog["common"])),
        ("Software Engineering Year 2", PATH_SE, len(catalog["se_year2"])),
        ("Software Engineering Year 2 unique", PATH_SE, len(catalog["se_year2_unique"])),
        ("Network Computing and Security Year 2", PATH_NCS, len(catalog["ncs_year2"])),
        ("Network Computing and Security Year 2 unique", PATH_NCS, len(catalog["ncs_year2_unique"])),
        ("Software Engineering signature", PATH_SE, len(catalog["signatures"][PATH_SE])),
        ("Data Science and Artificial Intelligence signature", PATH_DSAI, len(catalog["signatures"][PATH_DSAI])),
        ("Cyber and Information Security signature", PATH_CYBER, len(catalog["signatures"][PATH_CYBER])),
    ]
    return pd.DataFrame(rows, columns=["Course Set", "Assigned Pathway", "Courses"])


def review_reason_counts(mapping: pd.DataFrame) -> pd.DataFrame:
    """Count manual-review reasons from the evidence text."""
    review = mapping.loc[mapping["Assigned Pathway"].eq(PATH_REVIEW), "Mapping Evidence"].astype(str)
    reasons = review.map(lambda text: text.split(":", 1)[0].strip())
    if reasons.empty:
        return pd.DataFrame(columns=["Reason", "Students"])
    return reasons.value_counts().rename_axis("Reason").reset_index(name="Students")


def export_study_plan_mapping(
    mapping: pd.DataFrame,
    evidence: pd.DataFrame,
    validation: pd.DataFrame,
    path: Path,
) -> None:
    """Write the mapping workbook without changing the student profile."""

    def write_workbook(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            mapping.to_excel(writer, sheet_name="Student Study Plan Mapping", index=False)
            validation.to_excel(writer, sheet_name="Mapping Validation", index=False)
            evidence.to_excel(writer, sheet_name="Mapping Evidence", index=False)
            _format_table(writer.book["Student Study Plan Mapping"], {
                "A": 16, "B": 24, "C": 42, "D": 42, "E": 16, "F": 40, "G": 72,
                "H": 20, "I": 24, "J": 26, "K": 32, "L": 42, "M": 42, "N": 42, "O": 42,
            }, integer_columns={"J", "K"}, wrap_columns={"C", "F", "G", "L", "M", "N", "O"})
            _format_table(writer.book["Mapping Validation"], {"A": 62, "B": 14, "C": 12}, integer_columns={"B"})
            _format_table(writer.book["Mapping Evidence"], {
                "A": 16, "B": 16, "C": 24, "D": 46, "E": 18, "F": 22, "G": 36,
            }, text_columns={"A", "B"})

    write_atomic(path, write_workbook)


def _assign_software_track(level: str | None, year2: dict[str, object], higher: dict[str, object], catalog: dict[str, object], completed: set[str]) -> dict[str, object]:
    if higher["status"] == "selected" and higher["label"] in {PATH_SE, PATH_DSAI}:
        pathway = str(higher["label"])
        strength = "completed" if year2["strength"] == "completed" or higher["strength"] == "completed" else "supporting"
        codes = tuple(dict.fromkeys((*year2["codes"], *higher["codes"])))
        return _assignment(
            pathway=pathway,
            stage=_stage_for_pathway(pathway, level, higher["codes"], catalog),
            method=METHOD_UNIQUE if strength == "completed" else METHOD_PATTERN,
            evidence=_course_evidence(f"{pathway} courses", codes),
            confidence=CONFIDENCE_HIGH if strength == "completed" else CONFIDENCE_MEDIUM,
            review=False,
            matched_count=_matched_count(pathway, level, completed, catalog),
            evidence_codes=codes,
        )
    if level in {LEVEL_ADVANCED, LEVEL_BACHELOR}:
        return _review(
            "Year 2 is Software Engineering, but Year 3 specialization is not determined",
            _course_evidence("Software Engineering Year 2 courses", year2["codes"]),
            catalog,
            completed,
        )
    return _assignment(
        pathway=PATH_SE,
        stage="Year 2",
        method=METHOD_UNIQUE if year2["strength"] == "completed" else METHOD_PATTERN,
        evidence=_course_evidence("Software Engineering Year 2 courses", year2["codes"]),
        confidence=CONFIDENCE_HIGH if year2["strength"] == "completed" else CONFIDENCE_MEDIUM,
        review=False,
        matched_count=_matched_count(PATH_SE, LEVEL_DIPLOMA, completed, catalog),
        evidence_codes=year2["codes"],
    )


def _assign_network_track(level: str | None, year2: dict[str, object], higher: dict[str, object], catalog: dict[str, object], completed: set[str]) -> dict[str, object]:
    codes = tuple(dict.fromkeys((*year2["codes"], *higher["codes"])))
    if level == LEVEL_DIPLOMA or (level is None and higher["status"] != "selected"):
        return _assignment(
            pathway=PATH_NCS,
            stage="Year 2",
            method=METHOD_UNIQUE if year2["strength"] == "completed" else METHOD_PATTERN,
            evidence=_course_evidence("Network Computing and Security Year 2 courses", year2["codes"]),
            confidence=CONFIDENCE_HIGH if year2["strength"] == "completed" else CONFIDENCE_MEDIUM,
            review=False,
            matched_count=_matched_count(PATH_NCS, LEVEL_DIPLOMA, completed, catalog),
            evidence_codes=year2["codes"],
        )
    stage = _stage_for_pathway(PATH_CYBER, level, higher["codes"], catalog)
    if higher["status"] == "selected":
        method = METHOD_UNIQUE if higher["strength"] == "completed" else METHOD_PATTERN
        evidence = _course_evidence("Cyber and Information Security courses", codes)
        confidence = CONFIDENCE_HIGH if higher["strength"] == "completed" else CONFIDENCE_MEDIUM
    else:
        method = METHOD_PATTERN
        evidence = "Network Computing and Security Year 2 continues only to Cyber and Information Security" + _suffix(year2["codes"])
        confidence = CONFIDENCE_MEDIUM
    return _assignment(
        pathway=PATH_CYBER,
        stage=stage,
        method=method,
        evidence=evidence,
        confidence=confidence,
        review=False,
        matched_count=_matched_count(PATH_CYBER, level if level in {LEVEL_ADVANCED, LEVEL_BACHELOR} else LEVEL_ADVANCED, completed, catalog),
        evidence_codes=codes,
    )


def _assign_explicit(
    explicit: str,
    level: str | None,
    year2: dict[str, object],
    higher: dict[str, object],
    catalog: dict[str, object],
    completed: set[str],
) -> dict[str, object]:
    if explicit == PATH_CYBER and level == LEVEL_DIPLOMA:
        return _review(
            "Explicit specialization is not on the study plan for the current level",
            "Historical specialization is Cyber and Information Security and the current level is Diploma",
            catalog,
            completed,
        )
    if explicit == PATH_DSAI and level == LEVEL_DIPLOMA and higher["status"] != "selected":
        return _review(
            "Explicit specialization is not on the study plan for the current level",
            "Historical specialization is Data Science and Artificial Intelligence and the current level is Diploma",
            catalog,
            completed,
        )
    pathway = explicit
    evidence = f"Historical specialization is {explicit}"
    if explicit == PATH_NCS and level in {LEVEL_ADVANCED, LEVEL_BACHELOR}:
        pathway = PATH_CYBER
        evidence = "Historical specialization is Network Computing and Security, which continues only to Cyber and Information Security"
    codes = tuple(dict.fromkeys((*year2["codes"], *higher["codes"])))
    return _assignment(
        pathway=pathway,
        stage=_stage_for_pathway(pathway, level, codes, catalog),
        method=METHOD_EXPLICIT,
        evidence=evidence + _suffix(codes),
        confidence=CONFIDENCE_HIGH if level is not None else CONFIDENCE_MEDIUM,
        review=False,
        matched_count=_matched_count(pathway, level, completed, catalog),
        evidence_codes=codes,
    )


def _select_track(
    labeled_sets: tuple[tuple[str, set[str]], ...],
    completed: set[str],
    supporting: set[str],
) -> dict[str, object]:
    completed_hits = {label: tuple(sorted(completed & codes)) for label, codes in labeled_sets if completed & codes}
    if len(completed_hits) > 1:
        return {"status": "conflict", "hits": completed_hits, "strength": "completed", "codes": (), "label": None}
    if len(completed_hits) == 1:
        label, codes = next(iter(completed_hits.items()))
        return {"status": "selected", "label": label, "codes": codes, "strength": "completed", "hits": completed_hits}
    support_hits = {label: tuple(sorted(supporting & codes)) for label, codes in labeled_sets if supporting & codes}
    if len(support_hits) > 1:
        return {"status": "conflict", "hits": support_hits, "strength": "supporting", "codes": (), "label": None}
    if len(support_hits) == 1:
        label, codes = next(iter(support_hits.items()))
        return {"status": "selected", "label": label, "codes": codes, "strength": "supporting", "hits": support_hits}
    return {"status": "none", "label": None, "codes": (), "strength": "none", "hits": {}}


def _tracks_conflict(year2: dict[str, object], higher: dict[str, object]) -> bool:
    if year2["status"] != "selected" or higher["status"] != "selected":
        return False
    if year2["label"] == PATH_SE and higher["label"] == PATH_CYBER:
        return True
    if year2["label"] == PATH_NCS and higher["label"] in {PATH_SE, PATH_DSAI}:
        return True
    return False


def _explicit_contradiction(explicit: str, year2: dict[str, object], higher: dict[str, object]) -> str | None:
    year2_label = year2["label"] if year2["status"] == "selected" else None
    higher_label = higher["label"] if higher["status"] == "selected" else None
    contradictory = False
    if explicit in {PATH_SE, PATH_DSAI} and (year2_label == PATH_NCS or higher_label == PATH_CYBER):
        contradictory = True
    if explicit == PATH_SE and higher_label == PATH_DSAI:
        contradictory = True
    if explicit == PATH_DSAI and higher_label == PATH_SE:
        contradictory = True
    if explicit in {PATH_NCS, PATH_CYBER} and (year2_label == PATH_SE or higher_label in {PATH_SE, PATH_DSAI}):
        contradictory = True
    if not contradictory:
        return None
    return "Historical specialization " + explicit + " conflicts with " + _conflict_evidence(year2, higher)


def _within_common_year(courses: set[str], catalog: dict[str, object]) -> bool:
    """True when every historical course is absent or belongs to Common Year 1."""
    if not courses:
        return True
    plan_courses = courses & catalog["all_cores"]
    if not plan_courses:
        return False
    return plan_courses <= catalog["common"]


def _stage_for_pathway(pathway: str, level: str | None, codes: tuple[str, ...], catalog: dict[str, object]) -> str:
    if pathway == PATH_COMMON:
        return "Year 1"
    if pathway == PATH_NCS:
        return "Year 2"
    if pathway == PATH_SE and level in {None, LEVEL_DIPLOMA} and not _codes_on_block(codes, catalog, pathway, LEVEL_ADVANCED, LEVEL_BACHELOR):
        return "Year 2"
    if level == LEVEL_BACHELOR or _codes_on_block(codes, catalog, pathway, LEVEL_BACHELOR):
        return "Year 4"
    if level == LEVEL_ADVANCED or _codes_on_block(codes, catalog, pathway, LEVEL_ADVANCED):
        return "Year 3"
    if pathway == PATH_SE:
        return "Year 2"
    return "Year 3"


def _codes_on_block(codes: tuple[str, ...], catalog: dict[str, object], pathway: str, *levels: str) -> bool:
    if pathway not in catalog["blocks"]:
        return False
    return bool(set(codes) & catalog["blocks"][pathway]) and any(
        _placement_level(code, catalog) in levels for code in codes
    )


def _placement_level(code: str, catalog: dict[str, object]) -> str:
    placements = catalog["placements"].get(code, [])
    if not placements:
        return ""
    return placements[-1]["Level"]


def _matched_count(pathway: str, level: str | None, completed: set[str], catalog: dict[str, object]) -> int:
    return len(completed & _pathway_cores(pathway, level, catalog))


def _pathway_cores(pathway: str, level: str | None, catalog: dict[str, object]) -> set[str]:
    if pathway == PATH_COMMON:
        return set(catalog["common"])
    if pathway == PATH_NCS:
        return set(catalog["common"]) | set(catalog["ncs_year2"])
    if pathway == PATH_SE and level in {None, LEVEL_DIPLOMA}:
        return set(catalog["common"]) | set(catalog["se_year2"])
    cores = set(catalog["common"])
    if pathway in {PATH_SE, PATH_DSAI}:
        cores |= set(catalog["se_year2"])
    if pathway == PATH_CYBER:
        cores |= set(catalog["ncs_year2"])
    if level == LEVEL_BACHELOR:
        cores |= set(catalog["blocks"].get(pathway, set()))
    elif pathway in HIGHER_PATHWAYS:
        cores |= _cores_for_level(pathway, LEVEL_ADVANCED, catalog)
    return cores


def _cores_for_level(pathway: str, level: str, catalog: dict[str, object]) -> set[str]:
    codes = set()
    for code, placements in catalog["placements"].items():
        if any(item["Pathway"] == pathway and item["Level"] == level for item in placements):
            codes.add(code)
    return codes


def _evidence_rows(
    *,
    student_code: str,
    decision: dict[str, object],
    completed: set[str],
    failed: set[str],
    withdrawn: set[str],
    repeated: set[str],
    catalog: dict[str, object],
) -> list[dict[str, object]]:
    relevant = _relevant_codes(decision, completed | failed | withdrawn, catalog)
    rows = []
    if decision["method"] == METHOD_EXPLICIT:
        rows.append({
            "Student Code": student_code,
            "Course Code": pd.NA,
            "Course Status": pd.NA,
            "Matched Study Plan Pathway": decision["pathway"],
            "Study Plan Year": decision["stage"],
            "Study Plan Level": pd.NA,
            "Evidence Type": METHOD_EXPLICIT,
        })
    for code in sorted(relevant):
        status = _course_status(code, completed, failed, withdrawn, repeated)
        placement = _preferred_placement(code, str(decision["pathway"]), catalog)
        rows.append({
            "Student Code": student_code,
            "Course Code": code,
            "Course Status": status,
            "Matched Study Plan Pathway": placement["Pathway"],
            "Study Plan Year": placement["Year"],
            "Study Plan Level": placement["Level"],
            "Evidence Type": _evidence_type(status),
        })
    return rows


def _relevant_codes(decision: dict[str, object], historical: set[str], catalog: dict[str, object]) -> set[str]:
    if decision["pathway"] == PATH_COMMON:
        return historical & catalog["common"]
    signatures = set(catalog["se_year2_unique"]) | set(catalog["ncs_year2_unique"])
    for codes in catalog["signatures"].values():
        signatures |= set(codes)
    return historical & signatures


def _preferred_placement(code: str, pathway: str, catalog: dict[str, object]) -> dict[str, str]:
    placements = catalog["placements"].get(code, [])
    source_names = {
        PATH_SE: {PATH_SE, SOURCE_SE_YEAR2},
        PATH_DSAI: {PATH_DSAI, SOURCE_SE_YEAR2},
        PATH_CYBER: {PATH_CYBER, SOURCE_NCS_YEAR2},
        PATH_NCS: {SOURCE_NCS_YEAR2, PATH_CYBER},
        PATH_COMMON: {PATH_COMMON},
    }.get(pathway, set())
    for placement in placements:
        if placement["Pathway"] in source_names:
            return placement
    if placements:
        return placements[0]
    return {"Pathway": pathway, "Year": "", "Level": ""}


def _course_status(code: str, completed: set[str], failed: set[str], withdrawn: set[str], repeated: set[str]) -> str:
    if code in completed:
        label = "Completed"
    elif code in failed:
        label = "Failed"
    elif code in withdrawn:
        label = "Withdrawn"
    else:
        label = "Historical"
    if code in repeated:
        label = f"{label}; Repeated"
    return label


def _evidence_type(status: str) -> str:
    if status.startswith("Completed"):
        return "Completed Pathway Course"
    if status.startswith("Failed"):
        return "Failed Pathway Course"
    if status.startswith("Withdrawn"):
        return "Withdrawn Pathway Course"
    return "Pathway Course"


def _review(reason: str, detail: str, catalog: dict[str, object], completed: set[str]) -> dict[str, object]:
    evidence = reason if not detail or detail == reason else f"{reason}: {detail}"
    return _assignment(
        pathway=PATH_REVIEW,
        stage="Not assigned",
        method=METHOD_AMBIGUOUS,
        evidence=evidence,
        confidence=CONFIDENCE_REVIEW,
        review=True,
        matched_count=len(completed & catalog["all_cores"]),
        evidence_codes=(),
    )


def _assignment(
    *,
    pathway: str,
    stage: str,
    method: str,
    evidence: str,
    confidence: str,
    review: bool,
    matched_count: int,
    evidence_codes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "pathway": pathway,
        "stage": stage,
        "method": method,
        "evidence": evidence,
        "confidence": confidence,
        "review": review,
        "matched_count": int(matched_count),
        "evidence_codes": evidence_codes,
    }


def _conflict_evidence(year2: dict[str, object], higher: dict[str, object]) -> str:
    parts = []
    for label, decision in (("Year 2", year2), ("Year 3+", higher)):
        hits = decision.get("hits") or {}
        if hits:
            rendered = "; ".join(f"{name} ({', '.join(codes)})" for name, codes in hits.items())
            parts.append(f"{label}: {rendered}")
    return " | ".join(parts)


def _course_evidence(label: str, codes: tuple[str, ...] | list[str]) -> str:
    return label + _suffix(tuple(codes))


def _suffix(codes: tuple[str, ...]) -> str:
    if not codes:
        return ""
    shown = list(codes[:12])
    text = ", ".join(shown)
    if len(codes) > len(shown):
        text += ", ..."
    return ": " + text


def _core_codes(cores: pd.DataFrame, pathway: str, level: str | None = None) -> set[str]:
    selected = cores.loc[cores["Specialization Path"].eq(pathway)]
    if level is not None:
        selected = selected.loc[selected["Level"].eq(level)]
    return set(selected["Course Code"].astype(str))


def _explicit_specialization(value: object) -> str | None:
    text = normalize_text(value)
    if text.casefold() in {"", MISSING_SPECIALIZATION.casefold(), "manual review", UNKNOWN_LEVEL.casefold()}:
        return None
    return EXPLICIT_SPECIALIZATIONS.get(text.casefold())


def _canonical_level(value: object) -> str | None:
    return PLAN_LEVELS.get(normalize_text(value).casefold())


def _display_course_list(value: object) -> object:
    if _missing(value):
        return pd.NA
    text = normalize_text(value)
    return text if text and text.casefold() != "nan" else pd.NA


def _check(check: str, result: int, *, fail: bool = False, review: bool = False) -> dict[str, object]:
    if fail:
        status = "FAIL"
    elif review:
        status = "REVIEW"
    else:
        status = "PASS"
    return {"Check": check, "Result": int(result), "Status": status}


def _normalize_label(value: object) -> object:
    if _missing(value):
        return pd.NA
    text = normalize_text(value)
    return text if text else pd.NA


def _display_token(value: object) -> str:
    if _missing(value):
        return ""
    return str(value).strip()


def _yes(value: object) -> bool:
    return normalize_text(value).casefold() == "yes"


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


STATUS_RESOLVED = "Resolved"
STATUS_AMBIGUOUS = "Still Ambiguous"
STATUS_CONFLICT = "True Conflict"
STATUS_INSUFFICIENT = "Insufficient Historical Evidence"
STATUS_SOURCE = "Source Data Review Required"

READINESS_FULL = "Full Pathway Known"
READINESS_PARTIAL = "Partial Pathway Known"
READINESS_UNKNOWN = "Pathway Unknown"

SAFE_FULL = "Full remaining-course calculation"
SAFE_COMMON = "Common courses only"
SAFE_NONE = "Not safe"

METHOD_EXCLUSIVE_CORE = "Exclusive Completed Core"
METHOD_EXCLUSIVE_SUPPORT = "Exclusive Failed or Withdrawn Core"
METHOD_EXCLUSIVE_ELECTIVE = "Exclusive Elective Pool"
METHOD_CONTINUATION = "Official Year 2 Continuation"
METHOD_YEAR3_FORK = "Unresolved Software Engineering Year 3 Fork"
METHOD_INCOMPATIBLE = "Incompatible Completed Courses"
METHOD_INSUFFICIENT = "Insufficient Historical Evidence"

POOL_PATHWAYS = {
    "SE Major Elective": PATH_SE,
    "DSAI Major Elective": PATH_DSAI,
    "Cyber Major Elective": PATH_CYBER,
}
PATH_DISPLAY_ORDER = (PATH_COMMON, PATH_SE, PATH_NCS, PATH_DSAI, PATH_CYBER)
RESOLUTION_SUMMARY_COLUMNS = [
    "Student Code",
    "Current Level",
    "Original Mapping",
    "Original Review Reason",
    "Possible Pathways",
    "Resolved Pathway",
    "Resolution Status",
    "Resolution Method",
    "Resolution Evidence",
    "Confidence",
    "Safe for Remaining-Course Logic",
    "Remaining-Course Readiness",
    "Year 2 Base Path",
]
CONFLICT_EVIDENCE_COLUMNS = [
    "Student Code",
    "Current Level",
    "Course Code",
    "Course Title",
    "Course Status",
    "Academic Year",
    "Semester",
    "Matched Pathway",
    "Study Plan Level",
    "Study Plan Year",
    "Course Type",
    "Requirement Type",
    "Evidence Strength",
    "Conflict Group",
]
YEAR3_EVIDENCE_COLUMNS = [
    "Student Code",
    "Current Level",
    "Course Code",
    "Course Title",
    "Course Status",
    "Academic Year",
    "Semester",
    "Matched Pathway",
    "Study Plan Level",
    "Study Plan Year",
    "Course Type",
    "Requirement Type",
    "Evidence Strength",
    "Why It Does Not Resolve Year 3",
]


def build_course_roles(plan: pd.DataFrame, pools: pd.DataFrame) -> dict[str, object]:
    """Classify each official course by whether it can identify one pathway.

    A core that is also an elective on another pathway does not, by itself,
    contradict that other pathway. Placeholder elective labels are ignored.
    """
    catalog = build_pathway_catalog(plan)
    normalized = normalize_study_plan_courses(plan)
    placeholders = set(
        normalized.loc[normalized["Is Elective"].map(_yes) & normalized["Course Code"].notna(), "Course Code"].astype(str)
    )
    pool_paths: dict[str, set[str]] = {}
    pool_titles: dict[str, str] = {}
    for record in pools.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        pathway = POOL_PATHWAYS.get(normalize_text(record.get("Elective Pool")))
        if _missing(code) or pathway is None:
            continue
        pool_paths.setdefault(str(code), set()).add(pathway)
        if str(code) not in pool_titles:
            pool_titles[str(code)] = normalize_text(record.get("Course Title"))
    placements = catalog["placements"]
    metadata = _plan_metadata(normalized, pools)
    roles: dict[str, dict[str, object]] = {}
    for code in placeholders:
        roles[code] = _role("placeholder", None, set(), _placement_for(code, placements, metadata), "Weak")
    for pathway in HIGHER_PATHWAYS:
        for code in catalog["signatures"][pathway]:
            others = pool_paths.get(code, set()) - {pathway}
            kind = "exclusive_core" if not others else "core_elective_overlap"
            strength = "Strong" if kind == "exclusive_core" else "Overlap"
            roles[code] = _role(kind, pathway, others, _placement_for(code, placements, metadata), strength)
    for pathway, source_codes, label in (
        (PATH_SE, catalog["se_year2_unique"], "exclusive_year2"),
        (PATH_NCS, catalog["ncs_year2_unique"], "exclusive_year2"),
    ):
        for code in source_codes:
            if code in roles:
                continue
            others = pool_paths.get(code, set()) - {pathway}
            kind = label if not others else "year2_elective_overlap"
            strength = "Strong" if kind == "exclusive_year2" else "Overlap"
            roles[code] = _role(kind, pathway, others, _placement_for(code, placements, metadata), strength)
    for code in catalog["common"]:
        if code not in roles:
            roles[code] = _role("common", PATH_COMMON, set(), _placement_for(code, placements, metadata), "Weak")
    for code in catalog["all_cores"]:
        if code in roles:
            continue
        roles[code] = _role("shared_course", None, pool_paths.get(code, set()), _placement_for(code, placements, metadata), "Weak")
    for code, pathways in pool_paths.items():
        if code in roles or code in placeholders:
            continue
        if len(pathways) == 1:
            pathway = next(iter(pathways))
            roles[code] = _role("exclusive_elective", pathway, set(), _pool_placement(pathway, pool_titles.get(code, code)), "Moderate")
        else:
            roles[code] = _role("shared_elective", None, pathways, _pool_placement("", pool_titles.get(code, code)), "Weak")
    return {"catalog": catalog, "roles": roles, "placeholders": placeholders}


def judge_pathway_evidence(
    *,
    level: object,
    completed: set[str],
    failed: set[str],
    withdrawn: set[str],
    roles: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Resolve one student, or keep more than one possible pathway."""
    canonical_level = _canonical_level(level)
    supporting = failed | withdrawn
    strong = _hits(completed, roles, {"exclusive_core"})
    year2 = _hits(completed, roles, {"exclusive_year2"})
    moderate_cores = _hits(supporting, roles, {"exclusive_core"})
    moderate_electives = _hits(completed, roles, {"exclusive_elective"})
    se2 = set(year2.get(PATH_SE, ()))
    ncs2 = set(year2.get(PATH_NCS, ()))
    incompatible = _incompatible_strong(strong, se2, ncs2)
    if incompatible:
        possible = _possible_from_evidence(strong, se2, ncs2)
        return _resolution(
            possible=possible,
            resolved=None,
            status=STATUS_CONFLICT,
            method=METHOD_INCOMPATIBLE,
            evidence="Completed courses belong to incompatible pathways: " + incompatible,
            confidence=CONFIDENCE_REVIEW,
            readiness=READINESS_UNKNOWN,
            safe=SAFE_NONE,
            year2_base=_year2_base(se2, ncs2),
            evidence_codes=tuple(sorted(set().union(*strong.values(), se2, ncs2))),
        )
    if len(strong) == 1:
        pathway = next(iter(strong))
        if _continuation_allows(pathway, se2, ncs2):
            return _resolution(
                possible=(pathway,),
                resolved=pathway,
                status=STATUS_RESOLVED,
                method=METHOD_EXCLUSIVE_CORE,
                evidence=_exclusive_evidence(pathway, strong[pathway], roles, completed | supporting),
                confidence=CONFIDENCE_HIGH,
                readiness=READINESS_FULL,
                safe=SAFE_FULL,
                year2_base=_year2_base(se2, ncs2),
                evidence_codes=strong[pathway],
            )
    moderate = {path: codes for path, codes in moderate_cores.items() if codes}
    if not strong and len(moderate) == 1:
        pathway = next(iter(moderate))
        if _continuation_allows(pathway, se2, ncs2) and not _contrary_elective(pathway, moderate_electives):
            return _resolution(
                possible=(pathway,),
                resolved=pathway,
                status=STATUS_RESOLVED,
                method=METHOD_EXCLUSIVE_SUPPORT,
                evidence=_course_evidence(f"Failed or withdrawn courses unique to {pathway}", moderate[pathway]),
                confidence=CONFIDENCE_MEDIUM,
                readiness=READINESS_FULL,
                safe=SAFE_FULL,
                year2_base=_year2_base(se2, ncs2),
                evidence_codes=moderate[pathway],
            )
    if not strong and not moderate and len(moderate_electives) == 1:
        pathway = next(iter(moderate_electives))
        if _continuation_allows(pathway, se2, ncs2):
            return _resolution(
                possible=(pathway,),
                resolved=pathway,
                status=STATUS_RESOLVED,
                method=METHOD_EXCLUSIVE_ELECTIVE,
                evidence=_course_evidence(f"Elective pool contains {pathway} only", moderate_electives[pathway]),
                confidence=CONFIDENCE_MEDIUM,
                readiness=READINESS_FULL,
                safe=SAFE_FULL,
                year2_base=_year2_base(se2, ncs2),
                evidence_codes=moderate_electives[pathway],
            )
    if ncs2 and not se2 and not strong:
        pathway = PATH_NCS if canonical_level == LEVEL_DIPLOMA else PATH_CYBER
        stage_note = "Diploma remains Network Computing and Security Year 2" if pathway == PATH_NCS else (
            "Network Computing and Security Year 2 continues only to Cyber and Information Security"
        )
        return _resolution(
            possible=(pathway,),
            resolved=pathway,
            status=STATUS_RESOLVED,
            method=METHOD_CONTINUATION,
            evidence=stage_note + _suffix(tuple(sorted(ncs2))),
            confidence=CONFIDENCE_MEDIUM,
            readiness=READINESS_FULL,
            safe=SAFE_FULL,
            year2_base=PATH_NCS,
            evidence_codes=tuple(sorted(ncs2)),
        )
    if se2 and not ncs2:
        return _resolution(
            possible=(PATH_SE, PATH_DSAI),
            resolved=None,
            status=STATUS_AMBIGUOUS,
            method=METHOD_YEAR3_FORK,
            evidence="Software Engineering Year 2 is established. No exclusive Year 3 core, failed core, withdrawn core, or exclusive elective separates Software Engineering from Data Science and Artificial Intelligence",
            confidence=CONFIDENCE_REVIEW,
            readiness=READINESS_PARTIAL,
            safe=SAFE_COMMON,
            year2_base=PATH_SE,
            evidence_codes=tuple(sorted(se2)),
        )
    if not se2 and not ncs2 and not strong and not moderate and not moderate_electives:
        return _resolution(
            possible=(),
            resolved=None,
            status=STATUS_INSUFFICIENT,
            method=METHOD_INSUFFICIENT,
            evidence="No pathway-specific historical course evidence",
            confidence=CONFIDENCE_REVIEW,
            readiness=READINESS_UNKNOWN,
            safe=SAFE_NONE,
            year2_base=None,
            evidence_codes=(),
        )
    return _resolution(
        possible=_possible_from_evidence(strong or moderate or moderate_electives, se2, ncs2),
        resolved=None,
        status=STATUS_SOURCE,
        method=METHOD_INSUFFICIENT,
        evidence="Historical courses do not meet one official continuation",
        confidence=CONFIDENCE_REVIEW,
        readiness=READINESS_UNKNOWN,
        safe=SAFE_NONE,
        year2_base=_year2_base(se2, ncs2),
        evidence_codes=(),
    )


def build_mapping_resolution(
    mapping: pd.DataFrame,
    plan: pd.DataFrame,
    pools: pd.DataFrame,
    transcript: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Audit every manual-review student and resolve one only with exclusive evidence."""
    leakage = int(transcript["Semester"].astype(str).str.contains(HOLDOUT_TOKEN).sum()) if "Semester" in transcript.columns else 0
    attempts = _historical_attempts(transcript)
    course_roles = build_course_roles(plan, pools)
    roles = course_roles["roles"]
    review = mapping.loc[mapping["Manual Review Required"].eq("Yes")].copy()
    summaries = []
    conflict_rows = []
    year3_rows = []
    decisions: dict[str, dict[str, object]] = {}
    pair_rows: list[tuple[str, str]] = []
    for record in review.to_dict(orient="records"):
        student = str(record["Student Code"])
        courses = attempts.get(student, {})
        completed, failed, withdrawn = _status_sets(courses)
        decision = judge_pathway_evidence(
            level=record.get("Current Level"),
            completed=completed,
            failed=failed,
            withdrawn=withdrawn,
            roles=roles,
        )
        decisions[student] = decision
        original_reason = str(record.get("Mapping Evidence", "")).split(":", 1)[0]
        if original_reason.startswith("Conflicting"):
            pair_rows.extend(_student_pairs(courses, roles))
        summaries.append({
            "Student Code": student,
            "Current Level": record.get("Current Level"),
            "Original Mapping": PATH_REVIEW,
            "Original Review Reason": original_reason,
            "Possible Pathways": _join_paths(decision["possible"]),
            "Resolved Pathway": decision["resolved"] or pd.NA,
            "Resolution Status": decision["status"],
            "Resolution Method": decision["method"],
            "Resolution Evidence": decision["evidence"],
            "Confidence": decision["confidence"],
            "Safe for Remaining-Course Logic": decision["safe"],
            "Remaining-Course Readiness": decision["readiness"],
            "Year 2 Base Path": decision["year2_base"] or pd.NA,
        })
        if original_reason.startswith("Conflicting"):
            conflict_rows.extend(_audit_rows(student, record.get("Current Level"), courses, roles, decision, sheet="conflict"))
        if original_reason.startswith("Year 2"):
            year3_rows.extend(_audit_rows(student, record.get("Current Level"), courses, roles, decision, sheet="year3"))
    summary = pd.DataFrame(summaries, columns=RESOLUTION_SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values("Student Code", kind="mergesort").reset_index(drop=True)
    conflict = pd.DataFrame(conflict_rows, columns=CONFLICT_EVIDENCE_COLUMNS)
    year3 = pd.DataFrame(year3_rows, columns=YEAR3_EVIDENCE_COLUMNS)
    for frame in (conflict, year3):
        if not frame.empty:
            frame.sort_values(["Student Code", "Course Code"], kind="mergesort", inplace=True)
            frame.reset_index(drop=True, inplace=True)
    validation = _resolution_validation(summary, mapping, leakage, pair_rows)
    return {
        "summary": summary,
        "conflict": conflict,
        "year3": year3,
        "validation": validation,
        "decisions": decisions,
        "spring_2026_leakage": leakage,
    }


def apply_pathway_resolutions(
    mapping: pd.DataFrame,
    evidence: pd.DataFrame,
    resolution: pd.DataFrame,
    profiles: pd.DataFrame,
    plan: pd.DataFrame,
    decisions: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update manual-review rows that now have one exclusive pathway.

    Students who were already mapped stay on that pathway. Unresolved students
    stay on manual review.
    """
    updated = mapping.copy()
    updated_evidence = evidence.copy()
    resolved = resolution.loc[resolution["Resolution Status"].eq(STATUS_RESOLVED)].copy()
    if resolved.empty:
        return updated, updated_evidence
    automatic = set(mapping.loc[mapping["Manual Review Required"].eq("No"), "Student Code"].astype(str))
    if set(resolved["Student Code"].astype(str)) & automatic:
        raise RuntimeError("Resolution attempted to change an automatic mapping.")
    catalog = build_pathway_catalog(plan)
    profile_rows = {str(row["Student Code"]): row for row in profiles.to_dict(orient="records")}
    evidence_parts = [updated_evidence.loc[~updated_evidence["Student Code"].astype(str).isin(set(resolved["Student Code"].astype(str)))]]
    for record in resolved.to_dict(orient="records"):
        student = str(record["Student Code"])
        profile = profile_rows[student]
        completed = parse_course_list(profile.get("Completed Courses"))
        pathway = str(record["Resolved Pathway"])
        level = _canonical_level(profile.get("Current Level"))
        decision_record = (decisions or {}).get(student, {})
        codes = tuple(decision_record.get("evidence_codes") or ())
        decision = _assignment(
            pathway=pathway,
            stage=_stage_for_pathway(pathway, level, codes, catalog),
            method=METHOD_UNIQUE if record["Resolution Method"] == METHOD_EXCLUSIVE_CORE else METHOD_PATTERN,
            evidence=str(record["Resolution Evidence"]),
            confidence=str(record["Confidence"]),
            review=False,
            matched_count=_matched_count(pathway, level, completed, catalog),
            evidence_codes=codes,
        )
        mask = updated["Student Code"].astype(str).eq(student)
        updated.loc[mask, "Assigned Pathway"] = pathway
        updated.loc[mask, "Pathway Stage"] = decision["stage"]
        updated.loc[mask, "Mapping Method"] = decision["method"]
        updated.loc[mask, "Mapping Evidence"] = decision["evidence"]
        updated.loc[mask, "Mapping Confidence"] = decision["confidence"]
        updated.loc[mask, "Manual Review Required"] = "No"
        updated.loc[mask, "Matched Pathway Courses Count"] = decision["matched_count"]
        evidence_parts.append(pd.DataFrame(_evidence_rows(
            student_code=student,
            decision=decision,
            completed=completed,
            failed=parse_course_list(profile.get("Failed Courses")),
            withdrawn=parse_course_list(profile.get("Withdrawn Courses")),
            repeated=parse_course_list(profile.get("Repeated Courses")),
            catalog=catalog,
        ), columns=EVIDENCE_COLUMNS))
    new_evidence = pd.concat(evidence_parts, ignore_index=True)
    if not new_evidence.empty:
        new_evidence = new_evidence.sort_values(["Student Code", "Course Code"], kind="mergesort").reset_index(drop=True)
    unchanged = mapping.loc[mapping["Manual Review Required"].eq("No"), ["Student Code", "Assigned Pathway"]]
    compare = unchanged.merge(updated[["Student Code", "Assigned Pathway"]], on="Student Code", suffixes=("_original", "_updated"))
    if not compare["Assigned Pathway_original"].eq(compare["Assigned Pathway_updated"]).all():
        raise RuntimeError("An automatic pathway changed during resolution.")
    return updated, new_evidence


def export_mapping_resolution(tables: Mapping[str, pd.DataFrame], path: Path) -> None:
    """Write the manual-review resolution workbook."""

    def write_workbook(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            tables["summary"].to_excel(writer, sheet_name="Resolution Summary", index=False)
            tables["conflict"].to_excel(writer, sheet_name="Conflict Evidence", index=False)
            tables["year3"].to_excel(writer, sheet_name="Year3 Ambiguity", index=False)
            tables["validation"].to_excel(writer, sheet_name="Resolution Validation", index=False)
            _format_table(writer.book["Resolution Summary"], {
                "A": 16, "B": 22, "C": 18, "D": 42, "E": 55, "F": 42, "G": 28, "H": 42, "I": 78, "J": 14, "K": 42, "L": 28, "M": 42,
            }, wrap_columns={"D", "E", "H", "I", "K"})
            _format_table(writer.book["Conflict Evidence"], {
                "A": 16, "B": 22, "C": 16, "D": 42, "E": 22, "F": 16, "G": 16, "H": 42, "I": 22, "J": 18, "K": 16, "L": 22, "M": 20, "N": 36,
            }, text_columns={"A", "C"})
            _format_table(writer.book["Year3 Ambiguity"], {
                "A": 16, "B": 22, "C": 16, "D": 42, "E": 22, "F": 16, "G": 16, "H": 42, "I": 22, "J": 18, "K": 16, "L": 22, "M": 20, "N": 78,
            }, text_columns={"A", "C"}, wrap_columns={"N"})
            _format_table(writer.book["Resolution Validation"], {"A": 78, "B": 18, "C": 12}, integer_columns={"B"})

    write_atomic(path, write_workbook)


def _historical_attempts(transcript: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    """Keep the strongest pre-2026 attempt for each student and course."""
    required = {"Student Code", "Semester", "Academic Year", "Course Code", "Course Name", "Is Passed", "Is Failed", "Is Withdrawn"}
    missing = required - set(transcript.columns)
    if missing:
        raise RuntimeError("Historical transcript is missing columns: " + ", ".join(sorted(missing)))
    work = transcript.loc[~transcript["Semester"].astype(str).str.contains(HOLDOUT_TOKEN)].copy()
    if "Academic Year" in work.columns:
        work = work.loc[work["Academic Year"].astype(str).str.contains(HOLDOUT_TOKEN).eq(False)].copy()
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for record in work.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code):
            continue
        student = str(record["Student Code"])
        status = _attempt_status(record)
        repeated = _flag(record.get("Is Repeated Due To Failure")) or _flag(record.get("Is Repeated For Improvement")) or _attempt_number(record.get("Attempt Number")) > 1
        current = grouped.setdefault(student, {}).get(str(code))
        candidate = {
            "status": status,
            "year": record.get("Academic Year"),
            "semester": normalize_text(record.get("Semester")),
            "title": normalize_text(record.get("Course Name")),
            "repeated": repeated,
        }
        if current is None or _status_rank(status) > _status_rank(str(current["status"])):
            if current is not None and current.get("repeated"):
                candidate["repeated"] = True
            grouped[student][str(code)] = candidate
        elif repeated:
            current["repeated"] = True
    return grouped


def _audit_rows(student: str, level: object, courses: dict[str, dict[str, object]], roles: dict[str, dict[str, object]], decision: dict[str, object], *, sheet: str) -> list[dict[str, object]]:
    rows = []
    for code in sorted(courses):
        role = roles.get(code)
        if role is None or role["kind"] in {"placeholder", "common"}:
            continue
        if sheet == "year3" and role["kind"] == "off_plan":
            continue
        course = courses[code]
        status = str(course["status"])
        if course.get("repeated") and "Repeated" not in status:
            status = f"{status}; Repeated"
        placement = role["placement"]
        group = _conflict_group(role, decision) if sheet == "conflict" else _year3_effect(role)
        row = {
            "Student Code": student,
            "Current Level": level,
            "Course Code": code,
            "Course Title": placement.get("Title") or course.get("title") or code,
            "Course Status": status,
            "Academic Year": course.get("year"),
            "Semester": course.get("semester"),
            "Matched Pathway": role.get("pathway") or placement.get("Pathway") or pd.NA,
            "Study Plan Level": placement.get("Level") or pd.NA,
            "Study Plan Year": placement.get("Year") or pd.NA,
            "Course Type": placement.get("Course Type") or pd.NA,
            "Requirement Type": placement.get("Requirement Type") or pd.NA,
            "Evidence Strength": role["strength"],
        }
        if sheet == "conflict":
            row["Conflict Group"] = group
        else:
            row["Why It Does Not Resolve Year 3"] = group
        rows.append(row)
    return rows


def _resolution_validation(summary: pd.DataFrame, mapping: pd.DataFrame, leakage: int, pair_rows: list[tuple[str, str]]) -> pd.DataFrame:
    counts = summary["Resolution Status"].value_counts() if not summary.empty else pd.Series(dtype=int)
    resolved = int(counts.get(STATUS_RESOLVED, 0))
    ambiguous = int(counts.get(STATUS_AMBIGUOUS, 0))
    conflict = int(counts.get(STATUS_CONFLICT, 0))
    insufficient = int(counts.get(STATUS_INSUFFICIENT, 0))
    source = int(counts.get(STATUS_SOURCE, 0))
    total = int(len(summary))
    accounted = resolved + ambiguous + conflict + insufficient + source
    readiness = summary["Remaining-Course Readiness"].value_counts() if not summary.empty else pd.Series(dtype=int)
    full_review = int(readiness.get(READINESS_FULL, 0))
    partial_review = int(readiness.get(READINESS_PARTIAL, 0))
    unknown_review = int(readiness.get(READINESS_UNKNOWN, 0))
    original_review = int(mapping["Manual Review Required"].eq("Yes").sum())
    automatic = int(mapping["Manual Review Required"].eq("No").sum())
    ambiguous_paths_ok = True
    if not summary.empty:
        expected = _join_paths((PATH_SE, PATH_DSAI))
        fork = summary.loc[summary["Resolution Status"].eq(STATUS_AMBIGUOUS) & summary["Year 2 Base Path"].eq(PATH_SE)]
        ambiguous_paths_ok = bool(fork["Possible Pathways"].eq(expected).all()) if not fork.empty else True
    pair_counts = _pair_counts(pair_rows)
    checks = [
        _check("Original manual review", total, fail=total != original_review),
        _check("Resolved after deeper evidence", resolved),
        _check("Still ambiguous", ambiguous, review=ambiguous > 0),
        _check("True conflicts", conflict, review=conflict > 0),
        _check("Insufficient historical evidence", insufficient, review=insufficient > 0),
        _check("Source-data review required", source, review=source > 0),
        _check("Resolution statuses add to original manual review", accounted, fail=accounted != total),
        _check("Safe for full remaining-course calculation", full_review),
        _check("Safe only for common-course calculation", partial_review, review=partial_review > 0),
        _check("Not safe for remaining-course calculation", unknown_review, review=unknown_review > 0),
        _check("Readiness adds to original manual review", full_review + partial_review + unknown_review, fail=full_review + partial_review + unknown_review != total),
        _check("Full pathway known including automatic mappings", automatic + full_review),
        _check("Partial pathway known", partial_review, review=partial_review > 0),
        _check("Pathway unknown", unknown_review, review=unknown_review > 0),
        _check("Population adds to all students", automatic + full_review + partial_review + unknown_review, fail=automatic + full_review + partial_review + unknown_review != len(mapping)),
        _check("Ambiguous Software Engineering fork lists only SE and DSAI", int(ambiguous_paths_ok), fail=not ambiguous_paths_ok),
        _check("Spring 2026 leakage in resolution transcript", leakage, fail=leakage != 0),
        _check("Official curriculum-version crosswalk rows", 0, review=True),
    ]
    for label, count, judgment in pair_counts:
        checks.append(_check(f"Conflict pair {label}: {judgment}", count, review=count > 0 and judgment == "True incompatible evidence"))
    return pd.DataFrame(checks, columns=["Check", "Result", "Status"])


def _student_pairs(courses: dict[str, dict[str, object]], roles: dict[str, dict[str, object]]) -> list[tuple[str, str]]:
    """Classify the pathway pairs present in one original conflict student."""
    completed = {code for code, course in courses.items() if course["status"] == "Completed"}
    flags = {
        "se2": _has_kind(completed, roles, "exclusive_year2", PATH_SE) or _has_kind(completed, roles, "year2_elective_overlap", PATH_SE),
        "se2_exclusive": _has_kind(completed, roles, "exclusive_year2", PATH_SE),
        "ncs2": _has_kind(completed, roles, "exclusive_year2", PATH_NCS),
        "se_exclusive": _has_kind(completed, roles, "exclusive_core", PATH_SE),
        "dsai_exclusive": _has_kind(completed, roles, "exclusive_core", PATH_DSAI),
        "cyber_exclusive": _has_kind(completed, roles, "exclusive_core", PATH_CYBER),
        "se_overlap": _has_kind(completed, roles, "core_elective_overlap", PATH_SE),
        "dsai_overlap": _has_kind(completed, roles, "core_elective_overlap", PATH_DSAI),
        "cyber_overlap": _has_kind(completed, roles, "core_elective_overlap", PATH_CYBER),
    }
    pairs = []
    if flags["se2_exclusive"] and flags["ncs2"]:
        pairs.append(("SE Year 2 + NCS Year 2", "True incompatible evidence"))
    if (flags["se2"] or flags["se_exclusive"] or flags["se_overlap"]) and (flags["dsai_exclusive"] or flags["dsai_overlap"]):
        if flags["se_exclusive"] and flags["dsai_exclusive"]:
            judgment = "True incompatible evidence"
        elif flags["dsai_exclusive"] and not flags["se_exclusive"]:
            judgment = "Expected progression"
        else:
            judgment = "Possible curriculum overlap"
        pairs.append(("SE + DSAI", judgment))
    if (flags["se2"] or flags["se_exclusive"] or flags["se_overlap"]) and (flags["cyber_exclusive"] or flags["cyber_overlap"]):
        se_is_specific = flags["se2_exclusive"] or flags["se_exclusive"]
        if flags["cyber_exclusive"] and se_is_specific:
            judgment = "True incompatible evidence"
        else:
            judgment = "Possible curriculum overlap"
        pairs.append(("SE + Cyber", judgment))
    if flags["ncs2"] and (flags["cyber_exclusive"] or flags["cyber_overlap"]):
        pairs.append(("NCS + Cyber", "Expected progression"))
    if flags["dsai_exclusive"] and flags["cyber_exclusive"]:
        pairs.append(("DSAI + Cyber", "True incompatible evidence"))
    return pairs


def _has_kind(courses: set[str], roles: dict[str, dict[str, object]], kind: str, pathway: str) -> bool:
    return any(roles.get(code, {}).get("kind") == kind and roles.get(code, {}).get("pathway") == pathway for code in courses)


def _pair_counts(pair_rows: list[tuple[str, str]]) -> list[tuple[str, int, str]]:
    labels = (
        "SE Year 2 + NCS Year 2",
        "SE + Cyber",
        "SE + DSAI",
        "NCS + Cyber",
        "DSAI + Cyber",
    )
    grouped: dict[tuple[str, str], int] = {}
    for item in pair_rows:
        grouped[item] = grouped.get(item, 0) + 1
    rows = []
    for label in labels:
        matches = [(judgment, count) for (name, judgment), count in grouped.items() if name == label]
        if not matches:
            rows.append((label, 0, "Not present"))
            continue
        for judgment, count in sorted(matches):
            rows.append((label, count, judgment))
    return rows


def _hits(courses: set[str], roles: dict[str, dict[str, object]], kinds: set[str]) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = {}
    for code in sorted(courses):
        role = roles.get(code)
        if role is None or role["kind"] not in kinds or role.get("pathway") is None:
            continue
        found.setdefault(str(role["pathway"]), []).append(code)
    return {path: tuple(codes) for path, codes in found.items()}


def _incompatible_strong(strong: dict[str, tuple[str, ...]], se2: set[str], ncs2: set[str]) -> str:
    parts = []
    specs = [path for path in HIGHER_PATHWAYS if strong.get(path)]
    if len(specs) > 1:
        parts.append("; ".join(f"{path} ({', '.join(strong[path])})" for path in specs))
    if se2 and ncs2:
        parts.append(f"Software Engineering Year 2 ({', '.join(sorted(se2))}); Network Computing and Security Year 2 ({', '.join(sorted(ncs2))})")
    if se2 and strong.get(PATH_CYBER):
        parts.append(f"Software Engineering Year 2 ({', '.join(sorted(se2))}); {PATH_CYBER} ({', '.join(strong[PATH_CYBER])})")
    if ncs2 and strong.get(PATH_SE):
        parts.append(f"Network Computing and Security Year 2 ({', '.join(sorted(ncs2))}); {PATH_SE} ({', '.join(strong[PATH_SE])})")
    if ncs2 and strong.get(PATH_DSAI):
        parts.append(f"Network Computing and Security Year 2 ({', '.join(sorted(ncs2))}); {PATH_DSAI} ({', '.join(strong[PATH_DSAI])})")
    return " | ".join(parts)


def _continuation_allows(pathway: str, se2: set[str], ncs2: set[str]) -> bool:
    if se2 and ncs2:
        return False
    if pathway in {PATH_SE, PATH_DSAI} and ncs2:
        return False
    if pathway == PATH_CYBER and se2:
        return False
    if pathway == PATH_NCS and se2:
        return False
    return True


def _contrary_elective(pathway: str, electives: dict[str, tuple[str, ...]]) -> bool:
    return any(path != pathway and codes for path, codes in electives.items())


def _possible_from_evidence(strong: dict[str, tuple[str, ...]], se2: set[str], ncs2: set[str]) -> tuple[str, ...]:
    found = set(strong)
    if se2:
        found.add(PATH_SE)
    if ncs2:
        found.add(PATH_NCS)
    return tuple(path for path in PATH_DISPLAY_ORDER if path in found)


def _year2_base(se2: set[str], ncs2: set[str]) -> str | None:
    if se2 and not ncs2:
        return PATH_SE
    if ncs2 and not se2:
        return PATH_NCS
    return None


def _exclusive_evidence(pathway: str, codes: tuple[str, ...], roles: dict[str, dict[str, object]], courses: set[str]) -> str:
    overlap = []
    for code in sorted(courses):
        role = roles.get(code)
        if role is None or role["kind"] not in {"core_elective_overlap", "year2_elective_overlap"}:
            continue
        if pathway in set(role.get("compatible") or ()):
            overlap.append(code)
    text = _course_evidence(f"Completed courses unique to {pathway}", codes)
    if overlap:
        text += ". Courses also published as electives on the assigned pathway: " + ", ".join(overlap[:12])
    return text


def _conflict_group(role: dict[str, object], decision: dict[str, object]) -> str:
    kind = str(role["kind"])
    if decision["status"] == STATUS_CONFLICT and kind in {"exclusive_core", "exclusive_year2"}:
        return "True incompatible evidence"
    if kind == "exclusive_year2":
        return "Expected progression" if decision["resolved"] not in {None, role.get("pathway")} else "Year 2 base"
    if kind in {"core_elective_overlap", "year2_elective_overlap", "shared_elective"}:
        return "Possible curriculum overlap"
    if kind == "exclusive_core" and decision["resolved"] == role.get("pathway"):
        return "Expected progression"
    if kind == "exclusive_elective":
        return "Possible curriculum overlap"
    if kind == "shared_course":
        return "Shared course"
    return "Possible curriculum overlap"


def _year3_effect(role: dict[str, object]) -> str:
    kind = str(role["kind"])
    if kind == "exclusive_year2" and role.get("pathway") == PATH_SE:
        return "Establishes the Software Engineering Year 2 base. Year 3 can be Software Engineering or Data Science and Artificial Intelligence"
    if kind == "shared_course":
        return "Shared by more than one official pathway"
    if kind in {"shared_elective", "core_elective_overlap", "year2_elective_overlap"}:
        return "Official elective or shared course. It does not belong to only one Year 3 pathway"
    if kind == "exclusive_elective":
        return "Exclusive elective was not present for this unresolved student"
    return "Does not identify one Year 3 specialization"


def _resolution(
    *,
    possible: tuple[str, ...],
    resolved: str | None,
    status: str,
    method: str,
    evidence: str,
    confidence: str,
    readiness: str,
    safe: str,
    year2_base: str | None,
    evidence_codes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "possible": possible,
        "resolved": resolved,
        "status": status,
        "method": method,
        "evidence": evidence,
        "confidence": confidence,
        "readiness": readiness,
        "safe": safe,
        "year2_base": year2_base,
        "evidence_codes": evidence_codes,
    }


def _role(kind: str, pathway: str | None, compatible: set[str], placement: dict[str, object], strength: str) -> dict[str, object]:
    return {"kind": kind, "pathway": pathway, "compatible": set(compatible), "placement": placement, "strength": strength}


def _plan_metadata(plan: pd.DataFrame, pools: pd.DataFrame) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for record in plan.to_dict(orient="records"):
        code = record.get("Course Code")
        if _missing(code) or str(code) in metadata:
            continue
        metadata[str(code)] = {
            "Title": normalize_text(record.get("Course Title")) or str(code),
            "Course Type": normalize_text(record.get("Course Type")),
            "Requirement Type": normalize_text(record.get("Requirement Type")),
        }
    for record in pools.to_dict(orient="records"):
        code = normalize_course_code(record.get("Course Code"))
        if _missing(code) or str(code) in metadata:
            continue
        metadata[str(code)] = {
            "Title": normalize_text(record.get("Course Title")) or str(code),
            "Course Type": normalize_text(record.get("Course Type")) or "Elective",
            "Requirement Type": normalize_text(record.get("Requirement Type")) or "Specialization",
        }
    return metadata


def _pool_placement(pathway: str, title: str) -> dict[str, object]:
    return {"Pathway": pathway, "Year": "", "Level": "", "Title": title, "Course Type": "Elective", "Requirement Type": "Specialization"}


def _placement_for(code: str, placements: dict[str, list[dict[str, str]]], metadata: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = placements.get(code, [])
    info = metadata.get(code, {})
    if not rows:
        return {
            "Pathway": "",
            "Year": "",
            "Level": "",
            "Title": info.get("Title") or code,
            "Course Type": info.get("Course Type") or "",
            "Requirement Type": info.get("Requirement Type") or "",
        }
    row = rows[-1]
    return {
        "Pathway": row["Pathway"],
        "Year": row["Year"],
        "Level": row["Level"],
        "Title": info.get("Title") or code,
        "Course Type": info.get("Course Type") or "Core",
        "Requirement Type": info.get("Requirement Type") or "Core",
    }


def _join_paths(paths: tuple[str, ...]) -> str:
    ordered = [path for path in PATH_DISPLAY_ORDER if path in set(paths)]
    return " | ".join(ordered)


def _status_sets(courses: dict[str, dict[str, object]]) -> tuple[set[str], set[str], set[str]]:
    completed, failed, withdrawn = set(), set(), set()
    for code, course in courses.items():
        status = course["status"]
        if status == "Completed":
            completed.add(code)
        elif status == "Failed":
            failed.add(code)
        elif status == "Withdrawn":
            withdrawn.add(code)
    return completed, failed, withdrawn


def _attempt_status(record: dict[str, object]) -> str:
    if _flag(record.get("Is Passed")):
        return "Completed"
    if _flag(record.get("Is Failed")):
        return "Failed"
    if _flag(record.get("Is Withdrawn")):
        return "Withdrawn"
    return "Other"


def _status_rank(status: str) -> int:
    return {"Other": 0, "Withdrawn": 1, "Failed": 2, "Completed": 3}.get(status, 0)


def _attempt_number(value: object) -> int:
    if _missing(value):
        return 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if _missing(value):
        return False
    text = str(value).strip().casefold()
    return text in {"true", "yes", "y", "1"}


def _format_table(worksheet, widths: dict[str, int], *, integer_columns: set[str] | None = None, wrap_columns: set[str] | None = None, text_columns: set[str] | None = None) -> None:
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True)
    header_border = Border(bottom=Side(style="thin", color="666666"))
    for cell in worksheet[1]:
        cell.font = header_font
        cell.border = header_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    worksheet.freeze_panes = "B2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.row_dimensions[1].height = 22
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width
    integer_columns = integer_columns or set()
    wrap_columns = wrap_columns or set()
    text_columns = text_columns or set()
    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
        for cell in row:
            letter = get_column_letter(cell.column)
            if letter in text_columns and cell.value is not None:
                cell.number_format = "@"
            elif letter in integer_columns and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = "0"
            if letter in wrap_columns:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
