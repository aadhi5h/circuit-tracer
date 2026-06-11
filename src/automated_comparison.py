from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.patching import patch_activation
from src.compare import get_answer_logit_diff

def run_comparison(model, layer: int, correct_token: str = " Paris", incorrect_token: str = " London"):
    clean, corrupted = get_clean_corrupted_pair()

    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    corrupted_logits = model(corrupted_tokens)
    patched_logits = patch_activation(model, corrupted_tokens, clean_cache, "resid_post", layer=layer)

    return {
        "clean": get_answer_logit_diff(model, clean_logits, correct_token, incorrect_token),
        "corrupted": get_answer_logit_diff(model, corrupted_logits, correct_token, incorrect_token),
        "patched": get_answer_logit_diff(model, patched_logits, correct_token, incorrect_token),
    }

if __name__ == "__main__":
    model = load_model()
    for layer in range(model.cfg.n_layers):
        result = run_comparison(model, layer)
        print(f"layer {layer}: {result}")