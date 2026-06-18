# Project Discussion

## Strengths

- The project now has a reproducible script-based pipeline instead of only a notebook.
- Raw data, metadata, splits, metrics, reports, and figures are separated cleanly.
- The main evaluation split is subject-aware, preventing participant leakage.
- The original coursework models were reproduced under corrected evaluation before improvements were attempted.
- Diagnostics were run before adding the final SVM baseline, which kept the modelling process honest.
- Negative results were preserved rather than hidden, especially the weak CNN results.

## Limitations

- The dataset is small for an eight-class audio task: the corrected split has 560 training recordings.
- Each song has strong participant, recording type, pitch, tempo, and audio-quality variation.
- Feature extraction currently uses scipy-based approximations because `librosa` was not installed locally.
- The compact features summarize audio over time and therefore lose detailed melody ordering.
- The final Random Forest now saves a per-class test classification report and confusion matrix, but class-level conclusions should still be treated cautiously because the corrected test split is small.
- The CNN experiments were limited and should not be treated as a full deep learning study.

## Threats To Validity

- Validation and test sets are small, so model rankings may be unstable.
- Hyperparameter choices are validation-led, but validation subjects may not represent all participant variation.
- The file-level diagnostic split showed participant overlap but did not become easier for the selected RF setup; this suggests leakage effects depend on the exact overlap pattern and should not be generalized too broadly.
- The local feature implementation may differ numerically from the original notebook's `librosa` implementation.
- Some training augmentation is an approximation of the original notebook's augmentation.

## Future Work

- Use a robust audio library such as `librosa` consistently in a pinned environment.
- Add melody-aware features such as more reliable pitch contours, interval patterns, or dynamic time warping.
- Explore subject-aware cross-validation for more stable model estimates.
- Add calibration and confidence analysis.
- Revisit CNNs only with stronger regularization strategy, more data, transfer learning, or architecture search constrained by validation.
- Consider sequence models or template-matching approaches that preserve melodic ordering.
