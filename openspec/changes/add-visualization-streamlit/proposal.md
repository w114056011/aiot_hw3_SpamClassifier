## Why

Developers and graders need reproducible visual analysis of model performance and an interactive dashboard to explore results and run live inference during development. A Streamlit-based UI provides a fast, low-effort way to: (1) produce reproducible PNG/HTML reports for grading and (2) provide an interactive tool for manual inspection and live inference during demos.

## What Changes

- Add `add-visualization-streamlit` change containing:
  - A Streamlit application (`tools/streamlit_app.py`) that loads a trained model and vectorizer, provides batch and single-item inference, and visualizes evaluation artifacts.
  - A reproducible reporting script (`src/report.py`) to generate static reports (PNG/HTML) from `models/<model-name>/report.json` and model outputs.
  - Specs describing the visualization capability and the UI's behaviors.
  - Tasks and CI job that builds and exports a static report as a CI artifact.

**Breaking changes:** None. Additive developer tooling only.

## Impact

- Affected specs: adds `visualization` capability under changes.
- Affected code: new files under `tools/` and `src/` plus small docs and a sample Streamlit config. No production services changed.
- Rollout: Developer-only. Streamlit is optional at deploy; static reports are the deliverable for graders.

## Owner

- Proposed owner: course staff or repository maintainer. Please assign an owner for review.

## Timeline

- Spec & tasks scaffold: same day
- Implementation + tests + CI static report: 1 day

## Validation

- Run `openspec validate add-visualization-streamlit --strict` after adding spec deltas.
- Manual validation: run Streamlit locally (`streamlit run tools/streamlit_app.py`) and run `python src/report.py --model models/logreg --out reports/` to produce static artifacts.
