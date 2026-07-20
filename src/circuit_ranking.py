from itertools import combinations
from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.mlp_and_heads import patch_combo

COMPONENTS = {
    "L8H11": {"type": "head", "layer": 8, "head": 11},
    "L10H0": {"type": "head", "layer": 10, "head": 0},
    "MLP0": {"type": "mlp", "layer": 0},
}

def score_subset(model, corrupted_tokens, clean_cache, names: tuple):
    heads = [(COMPONENTS[n]["layer"], COMPONENTS[n]["head"]) for n in names if COMPONENTS[n]["type"] == "head"]
    mlps = [COMPONENTS[n]["layer"] for n in names if COMPONENTS[n]["type"] == "mlp"]
    logits = patch_combo(model, corrupted_tokens, clean_cache, heads, mlps)
    return get_answer_logit_diff(model, logits, " Paris", " London")

def rank_all_subsets(model):
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    names = list(COMPONENTS.keys())
    results = []
    for r in range(1, len(names) + 1):
        for subset in combinations(names, r):
            score = score_subset(model, corrupted_tokens, clean_cache, subset)
            results.append((subset, score))
    return sorted(results, key=lambda x: -x[1])

if __name__ == "__main__":
    model = load_model()
    ranked = rank_all_subsets(model)
    for subset, score in ranked:
        print(f"{' + '.join(subset)}: {score:.3f}")