from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.rank_heads import rank_all_heads

def test_top_head_causal_effect():
    model = load_model()
    scores = rank_all_heads(model)
    top_layer, top_head, top_diff = max(scores, key=lambda x: x[2])

    clean, corrupted = get_clean_corrupted_pair()
    corrupted_tokens = model.to_tokens(corrupted)
    corrupted_logits = model(corrupted_tokens)
    corrupted_diff = get_answer_logit_diff(model, corrupted_logits, " Paris", " London")

    assert top_diff > corrupted_diff, (
        f"expected top head patch to improve over corrupted baseline "
        f"({corrupted_diff:.3f}), got {top_diff:.3f}"
    )
    print(f"top causal head: L{top_layer}H{top_head}, diff {top_diff:.3f} (corrupted baseline {corrupted_diff:.3f})")

if __name__ == "__main__":
    test_top_head_causal_effect()