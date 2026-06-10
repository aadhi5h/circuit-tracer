from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.head_attribution import head_logit_attribution

def test_top_head_has_positive_attribution():
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    results = head_logit_attribution(model, clean, " Paris")
    top_label, top_score = max(results, key=lambda x: x[1])

    assert top_score > 0, f"expected positive attribution for top head, got {top_score:.3f}"
    print(f"top head {top_label} attribution: {top_score:.3f}")

if __name__ == "__main__":
    test_top_head_has_positive_attribution()