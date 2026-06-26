from src.model import load_model
from src.ioi_dataset import build_ioi_dataset
from src.investigate_candidate import investigate_head

def test_known_name_movers_show_strong_reverse_effect():
    """Reverse-patching (corrupted -> clean) should stay much closer to
    clean performance than to corrupted, for known name-mover heads —
    confirming they're part of a robust/redundant circuit rather than
    being individually decisive (which forward-patching alone can't show)."""
    model = load_model()
    examples = build_ioi_dataset()[:10]

    candidates = [(10, 0), (9, 9), (8, 6)]
    for layer, head in candidates:
        result = investigate_head(model, examples, layer, head)
        reverse = result["reverse_mean"]
        assert reverse > 1.0, (
            f"expected L{layer}H{head} reverse score > 1.0 (near-clean), "
            f"got {reverse:.3f}"
        )
        print(f"L{layer}H{head}: reverse={reverse:.3f} — pass")

if __name__ == "__main__":
    test_known_name_movers_show_strong_reverse_effect()