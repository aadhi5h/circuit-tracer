from src.model import load_model
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example
from src.sweep_runner import sweep_checkpoints

def test_layer_8_transition_holds_across_templates():
    model = load_model()
    examples = build_ioi_dataset()
    prefixes = ["When", "Then", "After"]

    for prefix in prefixes:
        ex = next(e for e in examples if e["prompt"].startswith(prefix))
        corrupted = corrupt_example(ex)
        results = sweep_checkpoints(model, ex["prompt"], corrupted["prompt"], ex["correct"], ex["incorrect"])

        recovery_by_label = {cp["label"]: cp["recovery"] for cp in results["checkpoints"]}
        layer6 = recovery_by_label["resid_post layer 6"]
        layer8 = recovery_by_label["resid_post layer 8"]

        assert layer8 > layer6 + 0.3, (
            f"template {prefix!r}: expected meaningful jump from layer 6 ({layer6:.1%}) "
            f"to layer 8 ({layer8:.1%})"
        )
        print(f"{prefix!r}: layer 6 -> layer 8 jump = {layer8 - layer6:+.1%} — pass")

if __name__ == "__main__":
    test_layer_8_transition_holds_across_templates()