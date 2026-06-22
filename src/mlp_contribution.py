from src.model import load_model
from src.prompts import get_clean_corrupted_pair

def mlp_logit_attribution(model, prompt: str, correct_token: str):
    logits, cache = model.run_with_cache(prompt)
    correct_id = model.to_single_token(correct_token)

    results = []
    for layer in range(model.cfg.n_layers):
        mlp_out = cache["mlp_out", layer][:, -1, :]  # [batch, d_model] at last token
        scaled = cache.apply_ln_to_stack(mlp_out.unsqueeze(0), layer=-1, pos_slice=-1).squeeze(0)
        attribution = (scaled @ model.W_U[:, correct_id]).item()
        results.append((layer, attribution))
    return results

if __name__ == "__main__":
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    results = mlp_logit_attribution(model, clean, " Paris")
    top5 = sorted(results, key=lambda x: -x[1])[:5]
    for layer, score in top5:
        print(f"layer {layer} mlp: attribution {score:.3f}")4