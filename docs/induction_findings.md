# Induction Head Findings - gpt2-small

## Method
Repeated a random token sequence (length 20, tokens repeated twice) and measured
each attention head's average weight on the "induction offset" - the position
(seq_len - 1) tokens back, which is where an induction head should look to
copy the pattern from the first occurrence.

## Result
Top heads by induction score:

| Layer | Head | Score |
|-------|------|-------|
| 7     | 10   | 0.889 |
| 5     | 5    | 0.886 |
| 5     | 1    | 0.886 |
| 6     | 9    | 0.844 |
| 7     | 2    | 0.804 |

## Notes
- L5H1, L5H5, and L6H9 match induction heads previously documented for
  gpt2-small in existing interpretability literature - confirms the
  measurement method is correctly identifying real induction behavior,
  not an artifact.
- All top 5 scores cluster in layers 5–7, consistent with induction being a
  mid-to-late-layer phenomenon in gpt2-small rather than early or final layers.
- Induction accuracy test (`tests/test_induction.py`) passes with max score
  0.889, well above the 0.3 threshold.

## Next
Use these identified heads (esp. L5H1, L5H5, L6H9) as candidates for
component-level attribution work in later experiments.