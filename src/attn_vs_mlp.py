from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff, full_hook_name
from src.rank_heads import rank_all_heads

def patch_mlp(model, corrupted_tokens, clean_cache, layer: int):
    def hook_fn(activation, hook):
        return clean_cache["mlp_out", layer]

    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(full_hook_name(layer, "mlp_out"), hook_fn)]
    )
    return logits

def rank_all_mlps(model, correct_token: str = " Paris", incorrect_token: str = " London"):
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    scores = []
    for layer in range(model.cfg.n_layers):
        logits = patch_mlp(model, corrupted_tokens, clean_cache, layer)
        diff = get_answer_logit_diff(model, logits, correct_token, incorrect_token)
        scores.append((layer, diff))
    return scores

if __name__ == "__main__":
    model = load_model()
    head_scores = rank_all_heads(model)
    mlp_scores = rank_all_mlps(model)

    print("top 3 heads:")
    for layer, head, diff in sorted(head_scores, key=lambda x: -x[2])[:3]:
        print(f"  L{layer}H{head}: {diff:.3f}")

    print("top 3 mlps:")
    for layer, diff in sorted(mlp_scores, key=lambda x: -x[1])[:3]:
        print(f"  layer {layer} mlp: {diff:.3f}")