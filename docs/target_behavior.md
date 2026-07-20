# Target Behavior: Indirect Object Identification (IOI)

## Task
Given a sentence with two names where one acts on the other
(e.g. "When John and Mary went to the store, John gave a drink to"),
the model must predict the name that did NOT perform the giving action
(Mary), not the one that did (John).

## Why this behavior
- Well-documented in existing interpretability literature (Wang et al.),
  giving a known circuit to validate against - same approach as the Day 6-7
  induction head cross-check.
- Clean binary metric: logit diff between correct name and incorrect name,
  directly reusable from existing `get_answer_logit_diff` utility.
- Involves multiple component types (name-mover heads, S-inhibition heads,
  duplicate-token heads) - richer than the single-fact Paris/London example.

## Success criteria
Model should show strong logit preference for the correct (indirect object)
name over the incorrect (subject) name across multiple sentence templates.