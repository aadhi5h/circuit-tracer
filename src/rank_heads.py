from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff

def patch_single_head(model, corrupted_tokens, clean_cache, layer: int, head: int):
    def hook_fn(activation, hook):
        activation[:, -1, head, :] = clean_cache["z", layer][:, -1, head, :]
        return activation

    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]
    )
    return logits

def rank_all_heads(model, correct_token: str = " Paris", incorrect_token: str = " London"):
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            logits = patch_single_head(model, corrupted_tokens, clean_cache, layer, head)
            diff = get_answer_logit_diff(model, logits, correct_token, incorrect_token)
            scores.append((layer, head, diff))
    return scores

if __name__ == "__main__":
    model = load_model()
    scores = rank_all_heads(model)
    top5 = sorted(scores, key=lambda x: -x[2])[:5]
    for layer, head, diff in top5:
        print(f"L{layer}H{head}: patched diff {diff:.3f}")