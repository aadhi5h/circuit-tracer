from src.model import load_model
from src.prompts import get_clean_corrupted_pair

def get_head_output_direction(model, cache, layer: int, head: int, position: int = -1):
    """Get this head's output vector (its contribution to the residual stream)."""
    z = cache["z", layer][0, position, head, :]  # [d_head]
    W_O = model.W_O[layer, head]  # [d_head, d_model]
    return z @ W_O  # [d_model]

def get_head_query_direction(model, cache, layer: int, head: int, position: int = -1):
    """Get what this head's query is reading from the residual stream at this position."""
    resid_pre = cache["resid_pre", layer][0, position, :]  # [d_model]
    return resid_pre

def composition_score(model, cache, layer_a: int, head_a: int, layer_b: int, head_b: int):
    """Rough measure of whether head_b's input includes head_a's output direction.
    Cosine similarity between head_a's output vector and the residual stream
    head_b reads from (only meaningful if layer_a < layer_b)."""
    import torch
    out_a = get_head_output_direction(model, cache, layer_a, head_a)
    in_b = get_head_query_direction(model, cache, layer_b, head_b)
    cos_sim = torch.nn.functional.cosine_similarity(out_a.unsqueeze(0), in_b.unsqueeze(0))
    return cos_sim.item()

if __name__ == "__main__":
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    _, cache = model.run_with_cache(clean)

    # L8H11 -> L10H0: does layer 10's head read layer 8's output?
    score = composition_score(model, cache, layer_a=8, head_a=11, layer_b=10, head_b=0)
    print(f"L8H11 -> L10H0 composition score (cosine sim): {score:.3f}")

    # sanity baseline: compare against an unrelated pair
    baseline_score = composition_score(model, cache, layer_a=8, head_a=11, layer_b=10, head_b=5)
    print(f"L8H11 -> L10H5 (control) composition score: {baseline_score:.3f}")