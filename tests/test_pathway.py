from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff

def patch_multiple_heads(model, corrupted_tokens, clean_cache, heads: list):
    """heads: list of (layer, head) tuples to patch simultaneously."""
    hooks = []
    heads_by_layer = {}
    for layer, head in heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, head_list in heads_by_layer.items():
        def make_hook(layer=layer, head_list=head_list):
            def hook_fn(activation, hook):
                for head in head_list:
                    activation[:, -1, head, :] = clean_cache["z", layer][:, -1, head, :]
                return activation
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook()))

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
    return logits

def test_combined_patch_beats_individual():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    both_logits = patch_multiple_heads(model, corrupted_tokens, clean_cache, [(8, 11), (10, 0)])
    both_diff = get_answer_logit_diff(model, both_logits, " Paris", " London")

    only_a_logits = patch_multiple_heads(model, corrupted_tokens, clean_cache, [(8, 11)])
    only_a_diff = get_answer_logit_diff(model, only_a_logits, " Paris", " London")

    only_b_logits = patch_multiple_heads(model, corrupted_tokens, clean_cache, [(10, 0)])
    only_b_diff = get_answer_logit_diff(model, only_b_logits, " Paris", " London")

    print(f"L8H11 only: {only_a_diff:.3f}")
    print(f"L10H0 only: {only_b_diff:.3f}")
    print(f"both together: {both_diff:.3f}")

    assert both_diff >= max(only_a_diff, only_b_diff), (
        f"expected combined patch ({both_diff:.3f}) to be at least as good as "
        f"the better individual patch ({max(only_a_diff, only_b_diff):.3f})"
    )

if __name__ == "__main__":
    test_combined_patch_beats_individual()