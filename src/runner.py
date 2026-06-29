from src.model import load_model
from src.utils import get_answer_logit_diff, full_hook_name

def patch_and_score(model, corrupted_tokens, clean_cache, example: dict,
                     hook_name: str, layer: int, head: int = None, position: int = -1):
    """Unified patch+score: works for full resid_post patches (head=None)
    or single-head patches (head given)."""
    def hook_fn(activation, hook):
        if head is None:
            activation[:, position, :] = clean_cache[hook_name, layer][:, position, :]
        else:
            activation[:, position, head, :] = clean_cache[hook_name, layer][:, position, head, :]
        return activation

    if head is None:
        hook_path = full_hook_name(layer, hook_name)
    else:
        hook_path = f"blocks.{layer}.attn.hook_z"

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_path, hook_fn)])
    return get_answer_logit_diff(model, logits, example["correct"], example["incorrect"])


def run_layer_sweep(model, example: dict, corrupted_prompt: str, hook_name: str = "resid_post"):
    _, clean_cache = model.run_with_cache(example["prompt"])
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    return [
        (layer, patch_and_score(model, corrupted_tokens, clean_cache, example, hook_name, layer))
        for layer in range(model.cfg.n_layers)
    ]


def run_head_sweep(model, example: dict, corrupted_prompt: str, hook_name: str = "z"):
    _, clean_cache = model.run_with_cache(example["prompt"])
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    return [
        (layer, head, patch_and_score(model, corrupted_tokens, clean_cache, example, hook_name, layer, head))
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]


if __name__ == "__main__":
    from src.prompts import get_clean_corrupted_pair
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    layer_results = run_layer_sweep(model, example, corrupted)
    print("layer sweep:", layer_results[:3], "...")

    head_results = run_head_sweep(model, example, corrupted)
    top3 = sorted(head_results, key=lambda x: -x[2])[:3]
    print("top 3 heads:", top3)