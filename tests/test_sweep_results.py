from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.sweep_runner import sweep_checkpoints

def test_paris_london_solved_at_embedding():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    results = sweep_checkpoints(model, clean, corrupted, " Paris", " London")
    embed_recovery = results["checkpoints"][0]["recovery"]
    assert embed_recovery > 0.95, f"expected near-full recovery at embed for Paris/London, got {embed_recovery:.1%}"

def test_ioi_not_solved_at_embedding():
    from src.ioi_dataset import build_ioi_dataset
    from src.ioi_corruption import corrupt_example

    model = load_model()
    ex = build_ioi_dataset()[0]
    corrupted = corrupt_example(ex)
    results = sweep_checkpoints(model, ex["prompt"], corrupted["prompt"], ex["correct"], ex["incorrect"])
    embed_recovery = results["checkpoints"][0]["recovery"]
    assert embed_recovery < 0.5, (
        f"expected IOI to require more than embedding alone, got {embed_recovery:.1%} recovery "
        f"(if this fails, IOI may be more embedding-solvable than assumed)"
    )
    print(f"IOI embed-only recovery: {embed_recovery:.1%} (confirms real computation is needed)")

if __name__ == "__main__":
    test_paris_london_solved_at_embedding()
    test_ioi_not_solved_at_embedding()