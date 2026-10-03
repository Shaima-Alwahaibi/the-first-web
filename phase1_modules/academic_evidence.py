"""Derived academic evidence taken from recorded transcript fields.

This module does not edit source workbooks and does not calculate a new CGPA.
"""

from __future__ import annotations

import pandas as pd

GPA_EXCLUDED = "Excluded"
GPA_INCLUDED = "Included"
PATTERN_BOTH = "Both GPA-eligible attempts remain counted"
PATTERN_NOT_CONSISTENT = "Not consistent"
EXCLUDED_GRADES = {"W", "OP", "NC"}


def classify_gpa_counting(transcript: pd.DataFrame) -> pd.DataFrame:
    """Keep the source grade point and record whether the source flag includes it in GPA."""

    frame = transcript.copy()
    frame["Derived GPA Inclusion"] = frame["Is GPA Counted"].map(_counted).map(
        lambda included: GPA_INCLUDED if included else GPA_EXCLUDED
    )
    frame["GPA Counting Evidence"] = "Is GPA Counted"
    return frame


def repeat_counting_pattern(transcript: pd.DataFrame) -> dict[str, object]:
    """Test whether every non-W, non-OP, non-NC attempt in a repeated course stays GPA-counted."""

    if transcript.empty:
        return {"pattern": PATTERN_BOTH, "groups": 0, "exceptions": 0}
    frame = transcript.copy()
    frame["_grade"] = frame["Grade"].map(lambda value: "" if pd.isna(value) else str(value).strip().upper())
    frame["_counted"] = frame["Is GPA Counted"].map(_counted)
    exceptions = 0
    groups = 0
    for _, part in frame.groupby(["Student Code", "Course Code"], dropna=False):
        if len(part) < 2:
            continue
        groups += 1
        eligible = part.loc[~part["_grade"].isin(EXCLUDED_GRADES | {""})]
        if eligible.empty:
            continue
        if not bool(eligible["_counted"].all()):
            exceptions += 1
    pattern = PATTERN_BOTH if exceptions == 0 else PATTERN_NOT_CONSISTENT
    return {"pattern": pattern, "groups": groups, "exceptions": exceptions}


def _counted(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "yes", "y", "1"}
