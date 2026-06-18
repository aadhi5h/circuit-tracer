from collections import defaultdict
from src.model import load_model
from src.ioi_dataset import build_ioi_dataset, TEMPLATES
from src.ioi_evaluate import evaluate_ioi

def metrics_by_template(results):
    by_template = defaultdict(list)
    for r in results:
        # match prompt back to its template by finding which template produced it
        for t in TEMPLATES:
            prefix = t.split("{A}")[0]
            if r["prompt"].startswith(prefix):
                by_template[t].append(r["logit_diff"])
                break

    summary = {}
    for template, diffs in by_template.items():
        summary[template] = {
            "n": len(diffs),
            "mean": sum(diffs) / len(diffs),
            "min": min(diffs),
            "max": max(diffs),
        }
    return summary

def accuracy(results):
    correct = sum(1 for r in results if r["logit_diff"] > 0)
    return correct / len(results)

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()
    results = evaluate_ioi(model, examples)

    print(f"overall accuracy (diff > 0): {accuracy(results):.3f}")
    print()
    for template, stats in metrics_by_template(results).items():
        print(f"{template!r}")
        print(f"  n={stats['n']} mean={stats['mean']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}")