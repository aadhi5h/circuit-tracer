# Methodology

## Activation patching

Core technique: run a "corrupted" prompt (behavior broken) but splice in
one specific activation from a "clean" run (behavior correct) at a single
hook point. If the corrupted output shifts back toward clean, that
component is causally implicated.

Two directions are used:
- **Forward patching** (clean → corrupted): does restoring this
  component's clean value help a broken run? Measures whether a
  component alone is *sufficient* to recover behavior.
- **Reverse patching** (corrupted → clean): does removing this
  component's clean value from a good run hurt it? Measures whether a
  component is *necessary* — often a better signal for genuinely
  load-bearing components, since forward-only patching can understate a
  head's importance if other components can compensate for it.

A head or MLP output can score weakly on forward patching but strongly
on reverse patching, or vice versa — always check both before ruling a
component out.

## Ablation

Alternative to patching: zero out a component's output entirely rather
than substituting a specific replacement value. Tests "does this
component matter at all" rather than "does this component's clean value
specifically matter." Used alongside patching as a cross-validation
method — genuine circuit components should show up as important under
both methods, even though the two don't produce identical rankings.

## Corruption design

Corruption must actually make the task *harder*, not just different.
Two corruption schemes were used:
- **Fact recall**: swap the answer entity (Paris → London).
- **IOI**: swap the repeated subject name for a random third name
  (ABC-corruption), changing which entity is the correct answer without
  resolving the sentence's underlying ambiguity.

A naive corruption (e.g. flipping an attractor noun's number to match
the subject) can accidentally *remove* the task's difficulty instead of
increasing it — always verify corrupted-vs-clean logit diff actually
drops before building further analysis on a corruption scheme.

## Position-specific patching

Patching an entire activation tensor (all sequence positions at once)
rather than a single position is a common bug that produces misleading
results: if clean and corrupted prompts are identical except at the
final token, full-tensor and single-position patching give identical
results, hiding the bug. The bug only surfaces when clean/corrupted
diverge mid-sequence (as in IOI). Always patch a specific position
(typically the last, right before prediction) unless deliberately
testing multiple positions.

## Layer aggregation

When summarizing per-head scores by layer, **mean** aggregation dilutes
signal badly — most heads in a layer have near-zero individual effect,
so averaging pulls every layer toward the same baseline. **Max**
aggregation per layer surfaces which layers contain standout components
far more clearly, since the real question is usually "does this layer
contain an important head," not "is this layer uniformly important."

## Composition analysis

Cosine similarity between one component's output direction and a later
component's residual-stream input is too crude to detect real
composition — the residual stream is dominated by the sum of every
other component's contribution, so this method showed no discriminative
power in practice. A rigorous approach requires path patching (isolating
one component's specific contribution to another's input), not
implemented in this project.

## Baseline sanity checks

Before attributing a behavior to specific components, check what
patching the token embedding alone recovers. If embedding-only patching
already achieves near-100% recovery, the task is solvable via lookup and
any "circuit" found downstream is incidental, not load-bearing. This
check should be the first step in any circuit investigation, not an
afterthought — it reframed a large portion of this project's fact-recall
findings after the fact.

## TransformerLens gotchas encountered

- `ActivationCache` keys use short hook names (e.g. `resid_post`), while
  `run_with_hooks` needs the full path (`blocks.{layer}.hook_resid_post`)
  — mixing these up produces a `KeyError` with a doubled `hook_hook_`
  prefix.
- `torch.diagonal`'s default dims are `(0, 1)`, not the last two axes —
  on a `[n_heads, seq, seq]` tensor this silently returns the wrong
  (often empty) diagonal. Index into a single head first, then take the
  diagonal of the resulting 2D tensor.
- Some named hooks (e.g. `hook_mlp_in`) exist in `model.hook_dict` but
  never fire unless a specific config flag is enabled — a hook that
  silently never executes produces no error, just unchanged output. Use
  `resid_mid` instead for pre-MLP residual stream access.
- Top-level hooks (`hook_embed`, `hook_pos_embed`) don't follow the
  `blocks.{layer}.hook_{name}` naming pattern used by per-block hooks —
  they need special-cased path construction and cache lookups without a
  layer argument.
- JSON serialization silently converts tuples to lists on round-trip —
  comparing loaded data with `==` against the original will fail even
  though the content is identical; convert back to tuples before
  comparing, or compare element-wise.