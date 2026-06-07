from src.model import load_model

def patch_activation(model, corrupted_tokens, clean_cache, hook_name: str, layer: int):
    def hook_fn(activation, hook):
        return clean_cache[hook_name, layer]

    full_hook_name = f"blocks.{layer}.hook_{hook_name}"
    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(full_hook_name, hook_fn)]
    )
    return logits

if __name__ == "__main__":
    from src.prompts import get_clean_corrupted_pair
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()

    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    logits = patch_activation(model, corrupted_tokens, clean_cache, "resid_post", layer=0)
    print(f"patched logits shape: {logits.shape}")