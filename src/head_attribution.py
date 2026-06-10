import torch
from src.model import load_model
from src.prompts import get_clean_corrupted_pair

def head_logit_attribution(model, prompt: str, correct_token: str):
    logits, cache = model.run_with_cache(prompt)
    correct_id = model.to_single_token(correct_token)

    # [n_layers * n_heads, batch, seq, d_model]
    per_head_resid, labels = cache.stack_head_results(layer=-1, return_labels=True)
    final_token_resid = per_head_resid[:, 0, -1, :]  # last token, all heads

    # apply final layernorm scale then project through unembed for the correct token
    scaled = cache.apply_ln_to_stack(final_token_resid.unsqueeze(1), layer=-1, pos_slice=-1).squeeze(1)
    attribution = scaled @ model.W_U[:, correct_id]

    return list(zip(labels, attribution.tolist()))

if __name__ == "__main__":
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    results = head_logit_attribution(model, clean, " Paris")
    top5 = sorted(results, key=lambda x: -x[1])[:5]
    for label, score in top5:
        print(f"{label}: attribution {score:.3f}")