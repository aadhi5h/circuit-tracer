from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.targeted_patching import patch_at_position

def summarize_paris_london(model):
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    last_pos = corrupted_tokens.shape[1] - 1

    clean_diff = get_answer_logit_diff(model, model(clean), " Paris", " London")
    corrupted_diff = get_answer_logit_diff(model, model(corrupted), " Paris", " London")

    checkpoints = [
        ("embed only", "embed", None),
        ("resid_pre layer 0 (embed+pos)", "resid_pre", 0),
        ("resid_mid layer 0 (+attn0)", "resid_mid", 0),
        ("mlp_out layer 0", "mlp_out", 0),
        ("resid_post layer 8", "resid_post", 8),
        ("resid_post layer 10", "resid_post", 10),
    ]

    print(f"clean: {clean_diff:.3f}  corrupted: {corrupted_diff:.3f}")
    print()
    for label, hook_name, layer in checkpoints:
        if layer is None:
            logits = patch_at_position(model, corrupted_tokens, clean_cache, hook_name, 0, last_pos)
        else:
            logits = patch_at_position(model, corrupted_tokens, clean_cache, hook_name, layer, last_pos)
        diff = get_answer_logit_diff(model, logits, " Paris", " London")
        recovery = (diff - corrupted_diff) / (clean_diff - corrupted_diff)
        print(f"{label}: {diff:.3f} ({recovery:.1%} recovery)")

if __name__ == "__main__":
    model = load_model()
    summarize_paris_london(model)