from src.model import load_model
from src.ioi_dataset import build_ioi_dataset, TEMPLATES
from src.ioi_corruption import corrupt_example
from src.sweep_runner import sweep_checkpoints

def get_examples_by_template(examples, template_prefix: str, n: int = 5):
    matches = [ex for ex in examples if ex["prompt"].startswith(template_prefix)]
    return matches[:n]

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()

    template_prefixes = {
        "When...drink": "When",
        "Then...ball": "Then",
        "After...keys": "After",
    }

    for label, prefix in template_prefixes.items():
        subset = get_examples_by_template(examples, prefix, n=5)
        print(f"=== {label} ===")
        accum = {}
        for ex in subset:
            corrupted = corrupt_example(ex)
            results = sweep_checkpoints(model, ex["prompt"], corrupted["prompt"], ex["correct"], ex["incorrect"])
            for cp in results["checkpoints"]:
                accum.setdefault(cp["label"], []).append(cp["recovery"])
        for cp_label, recoveries in accum.items():
            avg = sum(recoveries) / len(recoveries)
            print(f"  {cp_label}: {avg:.1%}")
        print()