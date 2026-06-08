from src.model import load_model
from src.induction import run_induction_experiment
from src.visualize_attention import get_induction_head_scores

def test_induction_heads_exist():
    model = load_model()
    seq_len = 20
    tokens, logits, cache = run_induction_experiment(model, seq_len)
    scores = get_induction_head_scores(model, cache, seq_len)

    max_score = max(scores.values())
    assert max_score > 0.3, f"expected at least one strong induction head, got max score {max_score:.3f}"
    print(f"induction head test passed, max score: {max_score:.3f}")

if __name__ == "__main__":
    test_induction_heads_exist()