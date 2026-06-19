import random
from src.ioi_dataset import NAMES

def corrupt_example(example: dict, seed: int = 0) -> dict:
    """ABC-corruption: replace the repeated subject name (2nd mention)
    with a random third name not already in the sentence."""
    prompt = example["prompt"]
    subject = example["incorrect"].strip()  # the name that gets repeated (subject)
    indirect = example["correct"].strip()

    rng = random.Random(seed + hash(prompt) % 10000)
    candidates = [n for n in NAMES if n not in (subject, indirect)]
    corrupt_name = rng.choice(candidates)

    # replace only the LAST occurrence of subject (the repeated mention)
    idx = prompt.rfind(subject)
    corrupted_prompt = prompt[:idx] + corrupt_name + prompt[idx + len(subject):]

    return {
        "prompt": corrupted_prompt,
        "correct": example["correct"],
        "incorrect": example["incorrect"],
        "corrupt_name": corrupt_name,
    }

if __name__ == "__main__":
    from src.ioi_dataset import build_ioi_dataset
    examples = build_ioi_dataset()
    for ex in examples[:3]:
        corrupted = corrupt_example(ex)
        print(f"clean:     {ex['prompt']!r}")
        print(f"corrupted: {corrupted['prompt']!r}")
        print()