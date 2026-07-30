from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example
from src.sweep_runner import sweep_checkpoints

def build_comparison_chart(model):
    clean, corrupted = get_clean_corrupted_pair()
    pl_results = sweep_checkpoints(model, clean, corrupted, " Paris", " London")

    ioi_ex = build_ioi_dataset()[0]
    ioi_corrupted = corrupt_example(ioi_ex)
    ioi_results = sweep_checkpoints(model, ioi_ex["prompt"], ioi_corrupted["prompt"],
                                      ioi_ex["correct"], ioi_ex["incorrect"])

    labels = [cp["label"] for cp in pl_results["checkpoints"]]
    pl_recovery = [cp["recovery"] for cp in pl_results["checkpoints"]]
    ioi_recovery = [cp["recovery"] for cp in ioi_results["checkpoints"]]

    return labels, pl_recovery, ioi_recovery

def print_side_by_side(labels, pl_recovery, ioi_recovery, width: int = 30):
    print(f"{'checkpoint':28s} {'Paris/London':>14s} {'IOI':>14s}")
    for label, pl, ioi in zip(labels, pl_recovery, ioi_recovery):
        pl_bar = "#" * max(0, int(min(pl, 1.0) * width))
        ioi_bar = "#" * max(0, int(min(ioi, 1.0) * width))
        print(f"{label:28s} {pl:>6.1%} {pl_bar:30s} {ioi:>6.1%} {ioi_bar}")

if __name__ == "__main__":
    model = load_model()
    labels, pl_recovery, ioi_recovery = build_comparison_chart(model)
    print_side_by_side(labels, pl_recovery, ioi_recovery)