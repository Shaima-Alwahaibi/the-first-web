# Intelligent Student Course Recommendation System
## Phase 1 – Rule-Based Recommendation Demo

One Streamlit page for the completed academic-rule phase. It reads a transcript, then shows the validated study-plan match, the full ranked list of eligible courses, and the semester bundle selected by the existing load rules. It does not train a model and it does not cut the eligible list down to four courses.

```text
Transcript PDF
      ↓
Extraction
      ↓
Student profile, study plan, remaining courses
      ↓
Prerequisites and advising rules
      ↓
Rule-based recommendation
```

Eligibility is applied before ranking. A blocked course scores 0.00 and is not selected. Spring 2026 stays the holdout and is not used to complete a course.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Choose **Run Sample Student** or upload a transcript PDF. The sample files are `STUD-175` and `STUD-016`.

When this folder is inside the research project, the original modules and validated outputs are used. A standalone clone uses `phase1_modules/` and `data/demo/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

From the research project root, the existing suite remains:

```bash
python -m unittest discover -s tests
```

## Privacy

Uploaded transcripts are kept for the current session only. Do not commit new transcripts, `.env` files, or downloaded recommendation files.

## Deployment

Repository: https://github.com/Shaima-Alwahaibi/the-first-web

Branch `main`. Main file `app.py`.

## Limitations

The page lists the unresolved Phase-1 items: the General Requirement pool, foundation evidence, Officially Postponed policy, and the missing level-mixing threshold. Later work covers grade prediction, collaborative filtering, content similarity, transformers, hybrid scoring, and CGPA prediction.
