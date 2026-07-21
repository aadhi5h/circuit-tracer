from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.circuit_ranking import score_subset

def test_full_circuit_reproduces_expected_recovery():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    full_circuit = ("L8H11", "L10H0", "MLP0")
    score = score_subset(model, corrupted_tokens, clean_cache, full_circuit)

    clean_diff = get_answer_logit_diff(model, model(clean), " Paris", " London")
    corrupted_diff = get_answer_logit_diff(model, model(corrupted), " Paris", " London")

    # circuit should recover meaningfully toward clean, though not necessarily all the way
    assert score > corrupted_diff, f"circuit patch ({score:.3f}) should beat corrupted baseline ({corrupted_diff:.3f})"
    assert score < clean_diff, f"circuit patch ({score:.3f}) recovering more than full clean ({clean_diff:.3f}) would be unexpected"

    recovery_fraction = (score - corrupted_diff) / (clean_diff - corrupted_diff)
    print(f"clean={clean_diff:.3f} corrupted={corrupted_diff:.3f} circuit_patch={score:.3f}")
    print(f"recovery fraction: {recovery_fraction:.1%}")

if __name__ == "__main__":
    test_full_circuit_reproduces_expected_recovery()