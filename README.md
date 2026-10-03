# Intelligent Student Course Recommendation System
## Phase 1 – Rule-Based Web Demonstration

This repository is the manager-facing demonstration of the completed Phase-1 academic pipeline for the UTAS student course recommendation research project. It shows how a transcript moves through extraction, validation, and the existing rule-based recommender.

## 1. Purpose

The application lets a research supervisor or department manager follow one student from a transcript PDF to a recommended study plan. The plan is the validated Phase-1 result. The page explains why a course is recommended or blocked.

## 2. Scope

The demo covers transcript processing, the academic profile, study-plan matching, remaining requirements, prerequisite checking, eligibility, and rule-based priority. It stops there.

## 3. Current Research Phase

Completed in this prototype:

- Transcript processing
- Academic profile
- Study-plan matching
- Remaining-course analysis
- Prerequisite checking
- Academic eligibility
- Rule-based recommendations

Future research, not implemented here:

- Grade prediction
- Collaborative filtering
- Content-based similarity
- Transformer models
- Hybrid recommendation
- CGPA impact prediction

## 4. Architecture

```text
Streamlit UI
      ↓
Transcript parser
      ↓
Input validator
      ↓
Pipeline adapter
      ↓
Existing Tasks 1–10 Python modules and their validated outputs
      ↓
Structured recommendation result
      ↓
Streamlit presentation and Excel export
```

The web layer does not reimplement passing grades, prerequisites, advising rules, or rule scores. Scores are read from `rule_based_recommender.py`. For a student already in the research cohort, the recommendation rows are the validated pre-Spring 2026 outputs. An unseen student code stops with a study-plan message instead of a guessed plan.

`phase1_modules/` contains unmodified copies of the research modules so a clean clone can import the rule engine. When this folder sits inside the original research project, the original modules are used.

## 5. Methodology

```text
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
```

Eligibility is applied before a rule score can place a course in the confirmed plan. Spring 2026 is the evaluation holdout. Rows from that term are shown when they appear on a transcript and are not used to complete a course or to train a model. This demo does not train anything.

## 6. Rule-Based Scoring

The score table is imported from the research module:

| Academic condition | Rule score |
|---|---:|
| Failed required | 1.00 |
| Withdrawn required | 0.90 |
| Pending previous-level core | 0.80 |
| Pending current-level core | 0.80 |
| Pending current-level elective | 0.60 |
| Higher-level course when mixing is allowed | 0.50 |
| Not academically eligible | 0.00 |

Previous-level and current-level cores both score 0.80 and remain separate priority classes. A blocked course is not selected.

## 7. Project Structure

```text
the first web/
├── app.py
├── README.md
├── requirements.txt
├── app/
│   ├── config.py
│   ├── transcript_parser.py
│   ├── transcript_validator.py
│   ├── pipeline_adapter.py
│   ├── recommendation_service.py
│   ├── export_service.py
│   ├── limitations.py
│   └── ui_helpers.py
├── phase1_modules/
├── data/
│   ├── samples/
│   ├── demo/
│   └── reference/
├── tests/
└── outputs/
```

## 8. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## 9. Local Run

From this folder:

```bash
streamlit run app.py
```

Use **Run Sample Student** to open `STUD-175` or `STUD-016`, review the extracted courses, then choose **Generate Recommendation**.

If the original research project is the parent directory, its validated outputs are used. Otherwise the bundled `data/demo` extract for the two sample students is used. Set `PHASE1_PROJECT_ROOT` only when the research project lives somewhere else.

## 10. Testing

From this folder:

```bash
python -m unittest discover -s tests -v
```

The research project’s own suite stays in the parent `tests/` directory and should still be run from the project root:

```bash
python -m unittest discover -s tests
```

## 11. Deployment

Streamlit Community Cloud can run this repository.

1. Push this folder to GitHub as `the-first-web`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Create an app from the repository, branch `main`, main file `app.py`.
4. The host installs `requirements.txt` and starts `app.py`.

No API keys are required. Do not add a secrets file for this demo.

## 12. Privacy

Uploaded transcripts are processed in the current session and are not intentionally stored. The repository does not contain the full student cohort, the raw transcript workbook, or the Spring 2026 holdout workbook. Sample files use coded student IDs. Do not commit new transcripts, `.env` files, or generated recommendation downloads.

## 13. Known Limitations

The in-app Limitations page reads the genuine unresolved items from the Phase-1 readiness report when that file is available. They include the missing General Requirement course list, missing FPMS0001 evidence, two attempts that cannot be reconstructed, the open Officially Postponed fail-count question, and the missing numeric LCGPA/English mixing threshold. The demo does not invent values for those items.

## 14. Future Research

Grade prediction, collaborative filtering, content-based similarity, transformer models, hybrid recommendation, and CGPA impact prediction are later research tasks. The result object reserves those score fields and leaves them empty.

## 15. GitHub repository information

Repository: [the-first-web](https://github.com/Shaima-Alwahaibi/the-first-web)

Branch: `main`

The repository is the deployment source. Sample transcripts and the two-student validated extract are included so the sample path runs after a clean clone. The full research dataset stays in the original project.
