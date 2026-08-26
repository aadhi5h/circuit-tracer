# Fact Recall Findings (Paris/London)

## Task

Single-fact completion: "The Eiffel Tower is located in the city
of ___" → "Paris". Corruption swaps the answer entity in the prompt
itself ("...of London").

## Initial investigation

Extensive patching, ablation, and composition analysis identified two
attention heads (L8H11, L10H0) and an early MLP layer (MLP0) as
apparently important components, cross-validated across multiple
independent methods:

- Both heads and the MLP showed measurable forward-patching effects.
- Ablation independently flagged overlapping heads as load-bearing.
- Combining all three components recovered ~80% of the clean-vs-corrupted
  behavioral gap.
- MLP0 appeared to dominate: subsets containing MLP0 scored far higher
  than subsets without it, regardless of which heads were included.

## Reframing: the task is solved at the embedding layer

Later, more targeted patching revealed that patching the token embedding
alone — before any transformer computation occurs — recovers **100%**
of the clean-vs-corrupted gap. Patching the residual stream immediately
before the first MLP layer also achieves full recovery, actually
*exceeding* what patching that MLP's own output achieves (74.7%),
implying the MLP's computation partially obscures rather than adds to
an already-complete signal.

This makes sense mechanistically: "Paris" and "London" are different
single BPE tokens, so their embeddings alone are trivially
distinguishable from the very first layer. The task is a lookup, not a
distributed computation — none of the earlier "circuit" work was wrong,
but it was measuring downstream readouts of a signal that was already
fully resolved upstream, not a genuine multi-component mechanism.

## Conclusion

This investigation was valuable for building and debugging the
patching/ablation infrastructure (and surfaced several real
implementation bugs — see `methodology.md`), but the task itself was a
poor example for demonstrating circuit-level computation. Always check
embedding-only recovery before treating downstream component-attribution
results as evidence of a genuine circuit; see IOI (`ioi_findings.md`)
for a task where this check actually validates deeper computation.