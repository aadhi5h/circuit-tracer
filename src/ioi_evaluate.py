from src.model import load_model
from src.utils import get_answer_logit_diff
from src.ioi_dataset import build_ioi_dataset

def evaluate_ioi(model, examples):
    results = []
    for ex in examples:
        logits = model(ex["prompt"])
        diff = get_answer_logit_diff(model, logits, ex["correct"], ex["incorrect"])
        results.append({**ex, "logit_diff": diff})
    return results

def summarize(results):
    diffs = [r["logit_diff"] for r in results]
    return {
        "n": len(diffs),
        "mean": sum(diffs) / len(diffs),
        "min": min(diffs),
        "max": max(diffs),
    }

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()
    results = evaluate_ioi(model, examples)
    summary = summarize(results)
    print(summary)

    worst3 = sorted(results, key=lambda r: r["logit_diff"])[:3]
    print("worst 3 examples:")
    for r in worst3:
        print(f"  {r['prompt']!r} -> diff {r['logit_diff']:.3f}")