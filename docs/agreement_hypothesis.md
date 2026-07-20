# Hypothesis: Subject-Verb Agreement Circuit

## Prediction
Based on the IOI findings (Days 20-24), where name-mover heads in layers
8-10 carried information forward to the final prediction, I predict:

1. A small number of "subject-tracking" heads will attend from the verb
   position back to the true subject noun (not the distractor), likely
   in a similar mid-to-late layer range (7-10) given gpt2-small's general
   pattern of resolving long-range dependencies in that region.
2. Early layers (0-3) will show weak/no causal effect, similar to the
   IOI layer-sweep pattern (Day 18) where layers 0-6 barely moved the
   needle and the real effect concentrated later.
3. Unlike IOI's forward-patching results (weak single-head effects, Day
   20-21), reverse-patching (Day 23-24 methodology) will more clearly
   reveal which heads are load-bearing versus redundant.

## What would falsify this
- If distractor-number information dominates the verb prediction (i.e.
  corruption of the distractor alone flips the answer as strongly as
  corrupting the actual subject), that suggests the circuit is doing
  surface-proximity matching rather than genuine subject-tracking -
  a known failure mode in real subject-verb agreement literature for
  smaller models.
- If effect is spread evenly across many heads/layers rather than
  concentrated, that would suggest gpt2-small solves this task
  differently than it solves IOI (more distributed, less a few
  "specialist" heads).