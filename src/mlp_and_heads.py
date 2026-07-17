from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff, full_hook_name

def patch_combo(model, corrupted_tokens, clean_cache, heads: list, mlp_layers: list):
    """Patch a combination of specific heads and specific MLP layers together,
    all at the last token position."""
    hooks = []
    heads_by_layer = {}
    for layer, head in heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, head_list in heads_by_layer.items():
        def make_head_hook(layer=layer, head_list=head_list):
            def hook_fn(activation, hook):
                for head in head_list:
                    activation[:, -1, head, :] = clean_cache["z", layer][:, -1, head, :]
                return activation
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_head_hook()))

    for layer in mlp_layers:
        def make_mlp_hook(layer=layer):
            def hook_fn(activation, hook):
                activation[:, -1, :] = clean_cache["mlp_out", layer][:, -1, :]
                return activation
            return hook_fn
        hooks.append((full_hook_name(layer, "mlp_out"), make_mlp_hook()))

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
    return logits

if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    configs = [
        ("heads only", [(8, 11), (10, 0)], []),
        ("mlp0 only", [], [0]),
        ("heads + mlp0", [(8, 11), (10, 0)], [0]),
    ]

    for name, heads, mlps in configs:
        logits = patch_combo(model, corrupted_tokens, clean_cache, heads, mlps)
        diff = get_answer_logit_diff(model, logits, " Paris", " London")
        print(f"{name}: {diff:.3f}")