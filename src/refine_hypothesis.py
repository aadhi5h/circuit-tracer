from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.mlp_and_heads import patch_combo

def score(model, corrupted_tokens, clean_cache, heads, mlps):
    logits = patch_combo(model, corrupted_tokens, clean_cache, heads, mlps)
    return get_answer_logit_diff(model, logits, " Paris", " London")

if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    clean_diff = get_answer_logit_diff(model, model(clean), " Paris", " London")
    corrupted_diff = get_answer_logit_diff(model, model(corrupted), " Paris", " London")

    configs = [
        ("original 3-component", [(8, 11), (10, 0)], [0]),
        ("+ L9H8 (Day 20 #3)", [(8, 11), (10, 0), (9, 8)], [0]),
    ]

    for name, heads, mlps in configs:
        s = score(model, corrupted_tokens, clean_cache, heads, mlps)
        recovery = (s - corrupted_diff) / (clean_diff - corrupted_diff)
        print(f"{name}: {s:.3f} ({recovery:.1%} recovery)")