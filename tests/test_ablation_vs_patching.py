from src.model import load_model
from src.utils import get_answer_logit_diff
from src.prompts import get_clean_corrupted_pair
from src.ablation import ablate_all_heads
from src.rank_heads import rank_all_heads

def test_ablation_and_patching_overlap():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    ablation_scores = ablate_all_heads(model, example)
    patching_scores = rank_all_heads(model)  # Day 10's Paris/London head ranking

    top10_ablation = set((l, h) for l, h, _ in sorted(ablation_scores, key=lambda x: x[2])[:10])
    top10_patching = set((l, h) for l, h, _ in sorted(patching_scores, key=lambda x: -x[2])[:10])

    overlap = top10_ablation & top10_patching
    print(f"top-10 ablation heads: {top10_ablation}")
    print(f"top-10 patching heads: {top10_patching}")
    print(f"overlap: {overlap} ({len(overlap)} heads)")

    # weak sanity check only - some overlap expected but not guaranteed to be large,
    # since the two methods measure related but distinct things
    assert len(overlap) >= 1, "expected at least 1 head to appear in both top-10 lists"

if __name__ == "__main__":
    test_ablation_and_patching_overlap()