# Candidate Circuit: Paris/London Fact Recall

## Components identified
Cross-validated via patching, ablation, and their
intersection:
- **L8H11** - attention head, layer 8
- **L10H0** - attention head, layer 10
- **MLP0** - MLP output, layer 0

## Pairwise interactions
- L8H11 + L10H0: roughly additive (combined gain ~1.140 vs naive sum ~1.136)
- L8H11 + MLP0: sub-additive (some shared/overlapping contribution)
- L10H0 + MLP0: sub-additive (same pattern)

## Full subset ranking
| Subset                  | Score  |
|--------------------------|--------|
| L8H11 + L10H0 + MLP0     | 1.311  |
| L8H11 + MLP0             | 1.265  |
| L10H0 + MLP0             | 1.055  |
| MLP0                     | 1.014  |
| L8H11 + L10H0            | -2.330 |
| L8H11                    | -2.800 |
| L10H0                    | -3.005 |

## Interpretation
The full 3-component combo wins, confirming the test's expectation - but
the ranking is dominated almost entirely by whether MLP0 is present.
Every subset containing MLP0 clusters near 1.0-1.3; every subset without
it clusters near -2.3 to -3.0. Adding both attention heads on top of MLP0
alone only improves the score by 0.297 (1.014 -> 1.311), while MLP0's
presence alone is worth 3.3-4.3 points regardless of pairing.

This suggests MLP0 is the dominant component in this candidate circuit,
with L8H11 and L10H0 playing a comparatively minor supporting role - not
three co-equal parts of a balanced circuit, as "circuit" framing might
imply.

## Caveats
- Solo/pairwise scores are all still short of clean's full logit diff
  (2.534) except where noted - this candidate circuit is a partial
  explanation, not the complete mechanism.
- Composition analysis using cosine similarity was inconclusive;
  a rigorous path-patching approach would be needed to confirm true
  information flow between components rather than just measuring combined
  patching effects.