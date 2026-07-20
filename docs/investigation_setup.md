# Investigation Setup: Subject-Verb Agreement

## Dataset
36 examples from `agreement_dataset.py`: 6 subjects (3 singular/plural
pairs: key/keys, author/authors, dog/dogs) x 6 distractor nouns
(cabinets/cabinet, shelves/shelf, houses/house), using template
"The {subject} near the {distractor}".

## Target metric
Logit diff between correct verb (agreeing with true subject) and incorrect
verb (agreeing with distractor), same methodology as IOI's
`get_answer_logit_diff`.

## Hypothesis
See `docs/agreement_hypothesis.md` - predicting mid-to-late layer
concentration (7-10) similar to IOI, with a falsification criterion around
whether the model does genuine subject-tracking vs. surface-proximity
matching to the distractor.

## Next steps
- Day 37: baseline evaluation test (confirm model shows expected preference
  before further pipeline work)
- Day 39: first causal tracing experiment on this task
- Day 40: rank candidate heads, compare against hypothesis