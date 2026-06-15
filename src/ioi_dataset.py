NAMES = ["John", "Mary", "Tom", "Alice", "Bob", "Sarah"]

TEMPLATES = [
    "When {A} and {B} went to the store, {A} gave a drink to",
    "Then {A} and {B} went to the park, and {A} gave the ball to",
    "After {A} and {B} left the office, {A} handed the keys to",
]

def build_ioi_dataset():
    examples = []
    for template in TEMPLATES:
        for i in range(len(NAMES)):
            for j in range(len(NAMES)):
                if i == j:
                    continue
                a, b = NAMES[i], NAMES[j]
                prompt = template.format(A=a, B=b)
                examples.append({
                    "prompt": prompt,
                    "correct": f" {b}",   # indirect object
                    "incorrect": f" {a}", # subject (wrong answer)
                })
    return examples

if __name__ == "__main__":
    examples = build_ioi_dataset()
    print(f"total examples: {len(examples)}")
    for ex in examples[:3]:
        print(ex)