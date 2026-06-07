# Activation Patching Workflow

## Setup
1. Define a clean prompt (correct behavior) and corrupted prompt (one token changed to break it).
2. Run clean prompt with `run_with_cache` to store all intermediate activations.
3. Run corrupted prompt normally to get baseline broken output.

## Patching
4. Run corrupted prompt again, but at one hook point, replace its activation
   with the clean run's activation at the same hook (`patch_activation`).
5. If output shifts back toward clean behavior, that hook/layer is causally
   implicated in producing the behavior.

## Metric
Use logit difference between the correct and incorrect answer token
(`get_answer_logit_diff`) as the comparison signal across clean / corrupted / patched runs.

## Naming note
`ActivationCache` keys use short hook names (e.g. `resid_post`), while
`run_with_hooks` needs the full path (`blocks.{layer}.hook_resid_post`).
`patch_activation` handles this translation internally.

## Constraint
Clean and corrupted prompts must tokenize to the same length for hook-level
patching to align correctly.