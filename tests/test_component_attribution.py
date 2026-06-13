from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.rank_heads import rank_all_heads
from src.attn_vs_mlp import rank_all_mlps

def test_best_component_beats_baseline():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    corrupted_tokens = model.to_tokens(corrupted)
    corrupted_logits = model(corrupted_tokens)
    corrupted_diff = get_answer_logit_diff(model, corrupted_logits, " Paris", " London")

    head_scores = rank_all_heads(model)
    mlp_scores = rank_all_mlps(model)

    best_head_diff = max(head_scores, key=lambda x: x[2])[2]
    best_mlp_diff = max(mlp_scores, key=lambda x: x[1])[1]
    best_overall = max(best_head_diff, best_mlp_diff)

    assert best_overall > corrupted_diff, (
        f"expected best single component to beat corrupted baseline "
        f"({corrupted_diff:.3f}), got {best_overall:.3f}"
    )
    print(f"best head: {best_head_diff:.3f}, best mlp: {best_mlp_diff:.3f}, baseline: {corrupted_diff:.3f}")

if __name__ == "__main__":
    test_best_component_beats_baseline()