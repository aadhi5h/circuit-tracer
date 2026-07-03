# Independent Circuit Investigation: Subject-Verb Agreement

## Task
Given a sentence with a subject noun and an intervening distractor noun of
different grammatical number (e.g. "The key to the cabinets"), the model
must predict the verb form agreeing with the true subject, not the nearer
distractor.

Example: "The key to the cabinets" -> "is" (agrees with singular "key"),
not "are" (agrees with plural "cabinets", which is closer in the sentence).

## Why this behavior
- Different failure mode than IOI: this is syntactic agreement under
  distraction, not entity/indirect-object tracking — likely engages
  different circuitry (attention to subject head-noun vs. surface-adjacent
  noun).
- Well-studied in NLP/interpretability literature (subject-verb agreement
  under attractor nouns), giving a cross-check point like IOI's name-mover
  heads.
- Reuses the entire existing pipeline (prompts, corruption, patching,
  ranking, serialization) built through Day 30 — only the dataset and
  target tokens change.

## Plan
1. Build a small dataset of subject/distractor/verb triples with varying
   number combinations (singular-subject+plural-distractor and vice versa).
2. Establish baseline: confirm gpt2-small actually shows the expected
   agreement preference before building on top of it.
3. Corruption: swap the distractor's number (plural <-> singular) while
   keeping the true subject the same, mirroring IOI's ABC-corruption logic.
4. Run layer/head patching via existing `runner.py`.
5. Compare found circuit against subject-verb agreement literature.