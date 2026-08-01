# Circuit Tracer - Final Research Writeup

## Overview
Investigated causal circuits in gpt2-small using TransformerLens on CPU,
across two tasks: a single-fact recall example (Paris/London) and Indirect
Object Identification (IOI), a well-documented interpretability benchmark.

## Key finding: task type determines where computation happens
The central result of this project (Day 55, 58): Paris/London and IOI
require fundamentally different amounts of transformer computation.

- **Paris/London**: 100% recovery from patching the token embedding alone.
  The task is solvable before any layer computes anything - a lookup, not
  a circuit. Later investigation (Days 43-52) into "candidate circuit"
  heads (L8H11, L10H0) and MLP0 was methodologically sound but was, in
  retrospect, chasing signal already present at the embedding layer.
- **IOI**: ~0-1.5% recovery through layer 6, then full resolution (101-107%)
  by layer 8, confirmed robust across 3 different sentence templates
  (Day 56) and multiple independent example sets (Day 20-21, 55, 59).
  This is a genuine multi-layer circuit.

## IOI circuit - validated findings
Final attribution graph (Day 59, 10-example average):

| Rank | Head  | Score |
|------|-------|-------|
| 1    | L8H6  | 0.951 |
| 2    | L9H9  | 0.766 |
| 3    | L8H10 | 0.534 |
| 4    | L7H9  | 0.434 |
| 5    | L7H3  | 0.112 |

- L8H6 and L9H9 consistently rank top-2 across every independent scan run
  (Day 20, 21, 59) despite different example subsets - strong evidence
  these are genuinely load-bearing, not sampling noise.
- L9H9 independently rediscovered as a name-mover head without prior
  knowledge of the published IOI circuit paper's results.
- L10H0, a commonly-cited name-mover head in the literature, initially
  appeared to contradict prior work (negative forward-patching score,
  Day 21) but reverse-patching (Day 23-24) showed it's genuinely
  load-bearing - the discrepancy was a methodology artifact (forward vs.
  reverse patching measure different things), not a real disagreement.
- Layer 6->8 transition confirmed robust across 3 sentence templates
  (Day 56): recovery jumps of +52 to +100 percentage points regardless
  of sentence structure, and reconfirmed here (Day 59) at +99.7pp on a
  fresh example.

## Methodological lessons
- Full-tensor patching (accidentally used through Day 17) versus
  single-position patching produce very different, sometimes identical-
  looking results - the bug was invisible on Paris/London (clean/corrupted
  differ only at the last token) and only surfaced on IOI (Day 18).
- Forward-patching (corrupted->clean) and reverse-patching (clean->
  corrupted) measure different things; a head can look unimportant by one
  method and clearly important by the other (Day 23).
- Layer aggregation method matters: mean-per-layer dilutes signal from
  concentrated effects; max-per-layer surfaces it (Day 26, 41).
- Component "importance" claims should be checked against the
  embedding-layer baseline before treating them as evidence of a genuine
  circuit - Day 54's finding significantly reframed 10 days of prior work
  on the Paris/London task.

## Conclusion
The IOI investigation (Days 13-24, 39-40, 56, 59) represents the project's
legitimate circuit-tracing result: a template-robust, multi-example-
validated, cross-method-confirmed causal pathway through gpt2-small's
middle-to-late layers, with L8H6 and L9H9 as the two most consistently
implicated heads. The Paris/London investigation, while valuable for
building and debugging the patching infrastructure, ultimately studied a
task that didn't require