# Classical Diagnostic Summary

## Best Validation Configuration

- Model: `rf`
- Duration: `10s`
- Feature set: `combined`
- Validation accuracy: `0.3583`
- Validation macro F1: `0.3537`

## Selected Test Performance

- Test accuracy: `0.2750`
- Test macro F1: `0.2629`
- Test weighted F1: `0.2594`

## Duration Finding

The best duration diagnostic row used `10s`, `rf`, and validation macro F1 `0.3537`.

## Feature Finding

The best feature diagnostic row used `combined` features, `rf`, and validation macro F1 `0.3537`.

## Split Finding

Subject-aware split rows:

```text
model split  accuracy  macro_f1 feature_set  duration_seconds
   rf   val  0.358333  0.353709    combined              10.0
   rf  test  0.275000  0.262876    combined              10.0
```

File-level diagnostic split rows, not main evaluation:

```text
model split  accuracy  macro_f1 feature_set  duration_seconds
   rf   val  0.225000  0.219151    combined              10.0
   rf  test  0.258333  0.247085    combined              10.0
```

This file-level diagnostic split allows participant overlap, but the selected RF summary-feature setup did not become easier in this run. This suggests the weak results are not explained by subject-aware splitting alone; the contaminated notebook results were likely inflated by the stronger overlap created when evaluating on datasets that duplicated training/validation files. The file-level split should not replace the subject-aware evaluation.

## Recommendation

Use the diagnostic-selected classical configuration as the next candidate baseline. Keep the smaller CNN scripts as experimental/negative-result code for now, and do not present the smaller CNN as an improvement.
