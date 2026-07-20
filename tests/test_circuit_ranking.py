from src.model import load_model
from src.circuit_ranking import rank_all_subsets

def test_full_circuit_is_best():
    model = load_model()
    ranked = rank_all_subsets(model)
    best_subset, best_score = ranked[0]

    assert set(best_subset) == {"L8H11", "L10H0", "MLP0"}, (
        f"expected full 3-component circuit to be best, got {best_subset} "
        f"with score {best_score:.3f}"
    )
    print(f"best subset confirmed: {best_subset} ({best_score:.3f})")

if __name__ == "__main__":
    test_full_circuit_is_best()