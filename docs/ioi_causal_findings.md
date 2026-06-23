# IOI Causal Findings — Preliminary

## Method
Single-head patching (last-position only) across all 144 gpt2-small heads,
averaged over N example sentences from the IOI dataset, using ABC-corruption
(subject's repeated mention swapped for a random third name).

## Result (5-example scan, Day 20)
| Rank | Head  | Score |
|------|-------|-------|
| 1    | L9H9  | 0.479 |
| 2    | L8H6  | 0.368 |
| 3    | L8H10 | 0.211 |
| 4    | L7H9  | -0.140 |
| 5    | L7H3  | -0.329 |

## Result (15-example scan, Day 21)
| Rank | Head   | Score  |
|------|--------|--------|
| 1    | L8H6   | 0.812  |
| 2    | L9H9   | 0.598  |
| 3    | L8H10  | 0.250  |
| 4    | L7H9   | 0.078  |
| 5    | L10H0  | -0.261 |
| 6    | L7H3   | -0.291 |
| 7    | L10H10 | -0.375 |
| 8    | L9H7   | -0.530 |
| 9    | L10H1  | -0.559 |

Score range: max=0.812, min=-2.149. Top head margin over #2: 0.215 (fairly tight).

## Validation
L9H9 and L8H6 both remain top-2 across the 5- and 15-example scans — rank
order shifted but neither dropped out, suggesting the top-tier heads are
real signal, not sample noise. L9H9 specifically is a well-documented
name-mover head in published IOI circuit analysis of gpt2-small.

L10H0, also a commonly-cited name-mover head in the literature, ranks #5
here with a *negative* score — this scan does not fully replicate the
published head-level ranking, even though it lands in the right layer
region (8-10) and independently surfaces L9H9. Worth investigating rather
than treating as full confirmation.

## Open questions
- Does the top-2 ordering (L8H6 vs L9H9) stabilize with the full 90-example
  dataset, or keep shifting?
- Why does L10H0 score negatively here despite being cited as a name-mover
  head elsewhere — corruption method difference, single-position-only
  patching, or something else?
- Several heads show negative scores (L10H10, L9H7, L10H1) — check whether
  these are suppression heads (actively pushing toward the wrong answer)
  rather than noise.