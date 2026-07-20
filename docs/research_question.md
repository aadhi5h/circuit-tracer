# Research Question: IOI Circuit in gpt2-small

## Question
Which components (attention heads, MLP layers) in gpt2-small are causally
responsible for correctly identifying the indirect object over the subject
in sentences with the "X and Y did Z, X did W to ___" structure?

## Approach
1. Establish baseline behavior across multiple templates/name pairs (done -
   see `tests/test_ioi_baseline.py`).
2. Build clean/corrupted pairs for this task (e.g. swap which name is
   indirect object, or use random name-swap corruption).
3. Apply layer-level and head-level activation patching (methods already
   built in Days 5, 9, 10) to this new dataset.
4. Compare found circuit against published IOI circuit findings
   (name-mover heads, S-inhibition heads, duplicate-token heads) as a
   validation cross-check, same pattern as the Day 6-7 induction head check.

## Baseline result
Average logit diff (correct - incorrect) on 20-example subset: 3.771