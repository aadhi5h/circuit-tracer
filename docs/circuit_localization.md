# Circuit Localization Method

## Layer-level sweep
`layer_sweep.py` patches each layer's `resid_post` individually from clean into
corrupted, tracking how much of the clean/corrupted logit-diff gap each layer
restores. Broad signal - tells you *which depth* matters.

## Head-level ranking
`rank_heads.py` goes finer: patches individual attention heads' `z` output
(pre-output-projection) one at a time, ranking all 144 heads (gpt2-small:
12 layers x 12 heads) by restored logit diff. Single-head patches are a much
smaller lever than full-layer patches - don't expect them to flip the sign
back to positive on their own; compare against the corrupted baseline instead
of zero.

## Cross-check
Top head by single-head patching (L8H11, diff -2.800 vs corrupted baseline
-3.470) is not the same as the top head by direct logit attribution (L11H3,
Day 8) - but it is the #2 head by attribution (0.424), so the two independent
methods agree it matters, just measuring different things: attribution
captures direct contribution to the output logit, while patching captures
how much restoring that head's output changes downstream computation.

## Visualization
`visualize_sweep.py` renders layer-level results as a text bar chart for
quick eyeballing without a plotting dependency.