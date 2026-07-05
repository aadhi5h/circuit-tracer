SUBJECTS = [
    ("key", "is", "are"),      # singular subject
    ("keys", "are", "is"),     # plural subject
    ("author", "is", "are"),
    ("authors", "are", "is"),
    ("dog", "is", "are"),
    ("dogs", "are", "is"),
]

DISTRACTORS = ["cabinets", "cabinet", "shelves", "shelf", "houses", "house"]

TEMPLATE = "The {subject} near the {distractor}"

def build_agreement_dataset():
    examples = []
    for subject, correct_verb, incorrect_verb in SUBJECTS:
        for distractor in DISTRACTORS:
            prompt = TEMPLATE.format(subject=subject, distractor=distractor)
            examples.append({
                "prompt": prompt,
                "correct": f" {correct_verb}",
                "incorrect": f" {incorrect_verb}",
                "subject": subject,
                "distractor": distractor,
            })
    return examples

if __name__ == "__main__":
    examples = build_agreement_dataset()
    print(f"total examples: {len(examples)}")
    for ex in examples[:3]:
        print(ex)