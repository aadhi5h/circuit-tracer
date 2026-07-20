# Subject-Verb Agreement - Causal Findings

## Corruption method
Subject number flip (e.g. "key" -> "keys"), keeping distractor fixed -
correct/incorrect verb labels swap accordingly. (Note: initial attempt
flipped the distractor instead, which accidentally made the task easier
rather than harder in the mismatched-number cases - corrected before
this scan.)

## Layer sweep (single example: "The key near the cabinets")
Gradual recovery from -4.070 (layer 0) to 1.422 (layer 11, matching clean),
with a sharp transition concentrated in layers 7-10 - closely mirroring
the IOI layer sweep pattern from Day 18.

## Head ranking (10-example average)
| Rank | Head   | Score  |
|------|--------|--------|
| 1    | L7H4   | -1.225 |
| 2    | L10H9  | -2.081 |
| 3    | L8H5   | -2.208 |
| 4    | L11H10 | -2.704 |
| 5    | L6H0   | -2.940 |
| 6    | L5H2   | -2.940 |
| 7    | L10H5  | -2.941 |
| 8    | L7H8   | -3.137 |
| 9    | L9H10  | -3.153 |
| 10   | L11H8  | -3.181 |

All scores negative (corrupted baseline: -4.179), consistent with IOI's
pattern (Day 20-21) where no single head fully restores clean behavior -
multiple components act together rather than any one being solely
responsible.

## Comparison to IOI
No overlap between this top-10 and IOI's top-3 (L9H9, L8H6, L8H10). This
argues against a single shared "general-purpose long-range retrieval"
mechanism - subject-verb agreement and indirect-object identification
appear to route through at least partially distinct heads, suggesting
gpt2-small's circuitry is more task-specific than task-general, at least
for these two behaviors.