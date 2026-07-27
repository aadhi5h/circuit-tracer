from src.model import load_model
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example
from src.sweep_runner import sweep_checkpoints, CHECKPOINTS

def aggregate_ioi_sweep(model, n_examples: int = 10):
    examples = build_ioi_dataset()[:n_examples]
    accum = {label: 0.0 for label, _, _ in CHECKPOINTS}

    for ex in examples:
        corrupted = corrupt_example(ex)
        results = sweep_checkpoints(model, ex["prompt"], corrupted["prompt"], ex["correct"], ex["incorrect"])
        for cp in results["checkpoints"]:
            accum[cp["label"]] += cp["recovery"] / n_examples

    return accum

if __name__ == "__main__":
    model = load_model()
    avg_recovery = aggregate_ioi_sweep(model, n_examples=10)
    print("average recovery by checkpoint (10 IOI examples):")
    for label, recovery in avg_recovery.items():
        print(f"  {label}: {recovery:.1%}")