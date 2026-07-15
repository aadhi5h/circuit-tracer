from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.rank_heads import rank_all_heads
from src.ablation import ablate_all_heads

def get_candidate_heads(model, example: dict, top_n: int = 10):
    """Narrow to heads that appear in top-N of BOTH patching and ablation
    rankings, plus a small margin set for heads strong in either alone."""
    patching_scores = rank_all_heads(model)  # uses Paris/London internally
    ablation_scores = ablate_all_heads(model, example)

    top_patching = set((l, h) for l, h, _ in sorted(patching_scores, key=lambda x: -x[2])[:top_n])
    top_ablation = set((l, h) for l, h, _ in sorted(ablation_scores, key=lambda x: x[2])[:top_n])

    confirmed = top_patching & top_ablation          # strong in both methods
    patching_only = top_patching - top_ablation
    ablation_only = top_ablation - top_patching

    return {
        "confirmed": confirmed,
        "patching_only": patching_only,
        "ablation_only": ablation_only,
    }

if __name__ == "__main__":
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    candidates = get_candidate_heads(model, example)
    print(f"confirmed (both methods, {len(candidates['confirmed'])} heads):")
    for layer, head in sorted(candidates["confirmed"]):
        print(f"  L{layer}H{head}")

    print(f"patching-only ({len(candidates['patching_only'])} heads):")
    for layer, head in sorted(candidates["patching_only"]):
        print(f"  L{layer}H{head}")

    print(f"ablation-only ({len(candidates['ablation_only'])} heads):")
    for layer, head in sorted(candidates["ablation_only"]):
        print(f"  L{layer}H{head}")