from src.model import load_model
from src.agreement_dataset import build_agreement_dataset
from src.agreement_corruption import corrupt_example
from src.runner import run_head_sweep

def scan_heads_averaged(model, examples: list):
    n = len(examples)
    accum = {}
    for ex in examples:
        corrupted = corrupt_example(ex)
        results = run_head_sweep(model, ex, corrupted["prompt"])
        for layer, head, diff in results:
            accum[(layer, head)] = accum.get((layer, head), 0) + diff / n
    return sorted(accum.items(), key=lambda x: -x[1])

if __name__ == "__main__":
    model = load_model()
    examples = build_agreement_dataset()[:10]  # subset for speed, matches IOI's Day 20 approach
    ranked = scan_heads_averaged(model, examples)
    print("top 10 heads (averaged over 10 examples):")
    for (layer, head), score in ranked[:10]:
        print(f"  L{layer}H{head}: {score:.3f}")