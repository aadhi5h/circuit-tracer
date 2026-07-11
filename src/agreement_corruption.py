from src.agreement_dataset import build_agreement_dataset

SUBJECT_PAIRS = {
    "key": "keys", "keys": "key",
    "author": "authors", "authors": "author",
    "dog": "dogs", "dogs": "dog",
}

def corrupt_example(example: dict) -> dict:
    """Flip the SUBJECT's number, keep distractor the same. This changes
    which verb is correct without resolving the subject/distractor number
    mismatch — unlike flipping the distractor, which can accidentally make
    the sentence easier instead of harder."""
    flipped_subject = SUBJECT_PAIRS[example["subject"]]
    corrupted_prompt = example["prompt"].replace(
        f"The {example['subject']} ", f"The {flipped_subject} "
    )
    return {
        "prompt": corrupted_prompt,
        "correct": example["incorrect"],   # subject flipped -> correct verb flips too
        "incorrect": example["correct"],
        "corrupted_subject": flipped_subject,
    }

if __name__ == "__main__":
    examples = build_agreement_dataset()
    for ex in examples[:3]:
        corrupted = corrupt_example(ex)
        print(f"clean:     {ex['prompt']!r} (correct: {ex['correct']!r})")
        print(f"corrupted: {corrupted['prompt']!r} (correct: {corrupted['correct']!r})")
        print()