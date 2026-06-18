# Final Model Selection

## Recommended Final Portfolio Model

The recommended final model is the Diagnostic Random Forest:

- Features: combined compact summaries of MFCC, MFCC deltas, log-mel, chroma/pitch, and spectral descriptors.
- Duration: 10 seconds.
- Split: corrected subject-aware train/validation/test split.
- Validation accuracy: `0.3583`.
- Validation macro F1: `0.3537`.
- Test accuracy: `0.2750`.
- Test macro F1: `0.2629`.

## Why This Model

The final model should be selected based on honest methodology, not simply the highest observed test score. Original Logistic Regression had the highest test macro F1 among the listed final-comparison models (`0.2878`), but selecting it now would rely on test-set hindsight. The diagnostic Random Forest was selected by validation performance after controlled diagnostics, then evaluated once on the test split.

The Random Forest is also defensible because it:

- Uses the corrected subject-aware split.
- Uses compact features that are more appropriate than flattened MFCCs for small-data classical ML.
- Achieved the best validation macro F1 among the controlled diagnostic configurations.
- Generalized better than the validation-selected SVM on test.
- Is interpretable enough for a portfolio report compared with a weak CNN.

## Models Not Selected

- Original Logistic Regression: strong test result, but not the validation-led final choice.
- Original Random Forest: weaker than diagnostic RF on validation and test.
- Original CNN: weak corrected test macro F1.
- Smaller experimental CNN: underfit and should be marked experimental.
- SVM: useful strong baseline, but test macro F1 was lower than diagnostic RF.

## Final Recommendation

Present Diagnostic Random Forest as the final portfolio model. Present Logistic Regression, Random Forest, CNN, and SVM as baselines/comparators, and explicitly explain that the corrected evaluation is the main contribution.
