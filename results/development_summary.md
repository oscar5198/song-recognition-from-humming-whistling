# Development Summary

## Original Project

The project began as a notebook prototype for classifying songs from humming or whistling recordings. The original approach trained Logistic Regression, Random Forest, and CNN models using audio features such as MFCCs and log-mel spectrograms.

## Problems Found

The audit found serious evaluation leakage:

- The 400-file dataset was duplicated inside the 800-file dataset.
- The notebook evaluated on data that overlapped with training/validation.
- Results reported as test performance were therefore inflated.
- The project lacked scripts, saved metrics, reproducible splits, and a clean structure.

## Methodology Fixes

The refactor created a reproducible project foundation:

- Moved exploratory work out of the portfolio-facing project root.
- Built `results/metadata.csv`.
- Built subject-aware `results/splits.csv`.
- Ensured no participant appears in more than one split.
- Added reusable feature extraction and training scripts.
- Saved metrics, reports, figures, and discussion artifacts.

## Experiments

Original corrected models were reproduced:

- Original Logistic Regression: test accuracy `0.2833`, macro F1 `0.2878`.
- Original Random Forest: test accuracy `0.2583`, macro F1 `0.2557`.
- Original CNN: test accuracy `0.2083`, macro F1 `0.1440`.

Improved/experimental models were attempted:

- Compact summary features were added.
- Random Forest improved modestly.
- Logistic Regression did not reliably improve.
- The smaller CNN underfit badly.

Diagnostics then tested:

- Durations of 10, 20, and 30 seconds.
- MFCC, mel, pitch/chroma, and combined features.
- A file-level diagnostic split labelled as not main evaluation.
- CNN underfitting/overfitting behavior.

Finally, SVM was added as one strong classical baseline:

- Best validation SVM: RBF kernel, `C=10`, `gamma=scale`, no class weighting.
- SVM test accuracy: `0.2333`, macro F1 `0.2337`.

## Final Model

The recommended final model is the Diagnostic Random Forest using 10-second combined compact features. It was selected by validation performance under the corrected subject-aware split, then evaluated once on the test split:

- Validation accuracy: `0.3583`.
- Validation macro F1: `0.3537`.
- Test accuracy: `0.2750`.
- Test macro F1: `0.2629`.

This is not the highest test score in isolation, but it is the most defensible final choice because the selection process was validation-led.

## Lessons Learned

- Correct data splitting matters more than model complexity.
- Leakage can make weak models look strong.
- Subject-aware evaluation gives a harder but more honest estimate.
- CNNs are not automatically better for small audio datasets.
- Compact handcrafted features plus classical models are realistic for this project scale.
- Honest negative results improve the credibility of the final portfolio report.

## Development Status

The project now has a reproducible evaluation pipeline, corrected splits, baselines, diagnostics, final model selection, and written analysis artifacts.
