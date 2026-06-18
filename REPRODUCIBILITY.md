# Reproducibility

This repository uses a fixed random seed (`42`) and a subject-aware train/validation/test split. The final portfolio model is the validation-selected Random Forest saved by `src/train_final_model.py`.

## Environment

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Data

Place the 800-file MLEndHWII audio sample in:

```text
data/MLEndHWII_sample_800/
```

The older 400-file sample is not used for corrected evaluation because it overlaps with the 800-file dataset.

## Pipeline

Run these commands from the repository root:

```bash
python src/build_metadata.py
python src/create_splits.py
python src/train_final_model.py
```

Optional baseline comparison:

```bash
python src/train_svm.py
```

## Expected Outputs

The final pipeline writes:

- `results/metadata.csv`
- `results/splits.csv`
- `models/final_random_forest.joblib`
- `results/final_random_forest_metrics.csv`
- `results/classification_reports/final_random_forest_test.txt`
- `figures/confusion_matrix_final_random_forest.png`

The checked-in summary artifacts for the portfolio are:

- `results/final_model_comparison.csv`
- `results/development_summary.md`
- `results/final_model_selection.md`
- `results/error_analysis.md`
- `results/project_discussion.md`
