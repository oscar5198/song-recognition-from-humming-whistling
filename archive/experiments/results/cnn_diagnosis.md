# CNN Diagnosis

The original CNN and smaller experimental CNN were checked using the saved metrics and curves.

## Saved Metrics
```text
       model split  accuracy  macro_precision  macro_recall  macro_f1  weighted_f1
original_cnn  test  0.208333         0.164536      0.210228  0.143955     0.144880
original_cnn   val  0.283333         0.321621      0.269954  0.196903     0.207882
improved_cnn  test  0.108333         0.046931      0.104027  0.056433     0.057987
improved_cnn   val  0.116667         0.059663      0.116758  0.066957     0.065316
```

## Interpretation

- The original CNN reached higher validation performance than the smaller experimental CNN, but still had weak corrected test performance.
- The smaller experimental CNN underfit: validation accuracy stayed near random chance and early stopping restored epoch 1 weights.
- The improved CNN architecture was probably too conservative for this dataset/feature representation, and normalization plus strong regularization did not rescue it.
- Training curves are saved at `figures/training_curves_original_cnn.png` and `figures/training_curves_improved_cnn.png`.

## Recommendation

Do not present the smaller experimental CNN as a final improvement. Keep it as a negative result, and revisit CNN design only after the classical feature diagnostics are understood.
