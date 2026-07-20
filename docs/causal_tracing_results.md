# Causal Tracing Results - Layer-Level Summary

## Method
Per-head patching scores aggregated by layer using
**max** across heads in that layer, visualized as a text bar chart.

## Result (mean aggregation)

layer 0 | ############### -0.840
layer 1 | ############### -0.840
layer 2 | ############### -0.841
layer 3 | ############### -0.840
layer 4 | ############### -0.839
layer 5 | ############### -0.839
layer 6 | ################ -0.832
layer 7 | ########################### -0.760
layer 8 | ######################################## -0.681
layer 9 | ######################## -0.781
layer 10 | ######### -0.879
layer 11 | -0.938

## Known limitation (mean aggregation)
Mean-per-layer heavily dilutes signal: most heads in any given layer have
near-zero individual effect, so averaging pulls every layer toward the
corrupted baseline (-0.884). Layer 8 still edged out as least negative,
consistent with it containing the strongest single head found so far
(L8H6, 0.812) - but the chart understated how concentrated the effect
actually is.

## Fix
Switched aggregation from mean to **max** per layer, since the question that
actually matters is "does this layer contain a standout head," not "is this
layer uniformly strong." This surfaces the true concentration of effect
much more clearly than the mean version above.

## Result (max aggregation - updated)
layer  0 |  -0.770
layer  1 |  -0.770
layer  2 |  -0.771
layer  3 |  -0.762
layer  4 |  -0.762
layer  5 |  -0.759
layer  6 | # -0.732
layer  7 | ##################### -0.121
layer  8 | ######################################## 0.430
layer  9 | ################################## 0.275
layer 10 | ########### -0.440
layer 11 | # -0.731

## Logging
Experiment results are now persisted to `experiments/logs/` as timestamped
JSON files via `src/logging_utils.py`, rather than only printed to stdout.