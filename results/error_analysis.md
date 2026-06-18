# Error Analysis

This analysis uses the saved corrected-evaluation metrics, classification reports, and confusion matrix figures. Final Random Forest artifacts remain in `results/` and `figures/`; older baseline and diagnostic artifacts are preserved under `archive/experiments/`. The main honest split is the subject-aware split in `results/splits.csv`.

## Overall Pattern

All models perform far below the original contaminated notebook results. This is expected after removing participant/file overlap from evaluation. The task is difficult because humming and whistling recordings vary strongly by participant, recording type, pitch range, tempo, microphone quality, and melodic memory.

## Easiest Songs

Across the saved test reports, the easiest songs are not perfectly consistent by model, but several patterns appear:

- Original Logistic Regression handled `Happy` best among the original baselines, with test F1 `0.45`.
- Original Logistic Regression also did comparatively better on `Friend` with F1 `0.38` and `Feeling` with F1 `0.32`.
- The SVM handled `Married`, `RememberMe`, `Happy`, and `Friend` best among its own classes, with test F1 scores around `0.31` to `0.35`.
- Original CNN mostly concentrated predictions into a small number of classes, but it identified `Necessities` better than most other classes, with test F1 `0.39`.

The easiest classes are therefore model-dependent. A cautious conclusion is that `Happy`, `Friend`, `Married`, and `Necessities` carry relatively stronger cues for at least some feature/model combinations.

## Hardest Songs

Hard classes also vary, but several are repeatedly weak:

- `NewYork` is especially weak for SVM: precision, recall, and F1 are all `0.00`.
- `TryEverything` is weak for original Logistic Regression, with test F1 `0.12`, and for SVM, with test F1 `0.14`.
- `RememberMe` and `TryEverything` collapse to F1 `0.00` for the original CNN.
- Original Random Forest struggles badly on `Happy` and `NewYork`, both with test F1 `0.07`.

These patterns suggest that the current handcrafted features do not reliably capture the melodic structure needed to distinguish several song pairs.

## Most Confused Song Pairs

From the saved test confusion matrix figures, now archived under `archive/experiments/figures/` for the earlier baseline runs:

- Original Logistic Regression often confuses `Married` as `Feeling` and `RememberMe` as `Married` or `NewYork`.
- Original Random Forest strongly confuses `Happy` as `TryEverything`, and also confuses `TryEverything` as `Married`.
- Original CNN collapses many predictions into `Married` and `Necessities`, including `Feeling -> Married`, `Happy -> Necessities`, `NewYork -> Married`, and `RememberMe -> Married`.
- SVM strongly confuses `Friend -> NewYork`, `Necessities -> Feeling`, and `NewYork -> Married`.

The confusion patterns are musically plausible: short hummed or whistled clips can share broad contour, rhythm fragments, or pitch ranges, and the current summary features discard detailed temporal ordering.

## Why Random Forest Performed Best

The diagnostic Random Forest was selected by validation macro F1 using compact combined features. It likely helped because:

- It can model nonlinear interactions between pitch/chroma, spectral, MFCC, and mel summary statistics.
- It is less sensitive than Logistic Regression and SVM to feature scaling choices and non-Gaussian feature distributions.
- It can use mixed feature groups without requiring a single smooth decision boundary.

Its advantage is still modest. Validation macro F1 was `0.3537`, but test macro F1 was only `0.2629`, showing that generalization remains fragile.

## Why SVM Did Not Outperform Random Forest

The SVM matched the diagnostic Random Forest on validation accuracy (`0.3583`) and came very close on validation macro F1 (`0.3507` versus RF `0.3537`). However, its test macro F1 dropped to `0.2337`.

Possible reasons:

- The selected RBF SVM may be more sensitive to the exact validation subjects than the Random Forest.
- The compact feature space may contain class boundaries that are irregular rather than smooth.
- The small dataset means validation selection is noisy.
- Several SVM configurations tied on validation, suggesting the validation signal was not strong enough to distinguish robust settings.

SVM remains a useful baseline, but it is not the strongest final candidate.

## Why CNN Struggled

The original CNN had weak corrected test performance: test accuracy `0.2083`, macro F1 `0.1440`. The smaller experimental CNN collapsed further: test accuracy `0.1083`, macro F1 `0.0564`.

Likely causes:

- Only 560 training examples are available after subject-aware splitting.
- Humming/whistling has high participant variability.
- The CNN may learn participant- or recording-specific cues under contaminated splits, but those cues do not transfer to unseen participants.
- The smaller experimental CNN underfit: training/validation curves show early stopping restored epoch 1 weights.
- Log-mel spectrograms preserve time-frequency structure, but the current CNN setup is not robust enough for the limited data size.

The CNN should be presented as an important negative result, not as the final model.

## Why Corrected Evaluation Reduced Performance

The original notebook used overlapping data: the 400-file dataset was duplicated inside the 800-file dataset, and test evaluation included data overlapping with training/validation. That leakage made the task look much easier than it is.

The corrected split groups by participant ID, so no participant appears in more than one split. This prevents the model from benefiting from participant-specific voice, microphone, or recording habits. The performance drop is therefore not a failure of the project; it is the central methodological finding.

## Key Lessons

- Data splitting can dominate apparent model quality.
- A lower but honest score is more valuable than a high leaked score.
- Compact features improved validation behavior, but not enough to solve the task.
- CNNs are not automatically better for small audio datasets.
- The final model should be selected by validation-led methodology, not by whichever test result happens to be highest.
