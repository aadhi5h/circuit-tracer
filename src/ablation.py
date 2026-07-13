from src.model import load_model
from src.utils import get_answer_logit_diff

def ablate_head(model, tokens, layer: int, head: int, example: dict, mode: str = "zero"):
    """Zero-ablate a single head's output at all positions."""
    def hook_fn(activation, hook):
        if mode == "zero":
            activation[:, :, head, :] = 0.0
        return activation

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]
    )
    return get_answer_logit_diff(model, logits, example["correct"], example["incorrect"])

def ablate_all_heads(model, example: dict):
    tokens = model.to_tokens(example["prompt"])
    scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            diff = ablate_head(model, tokens, layer, head, example)
            scores.append((layer, head, diff))
    return scores

if __name__ == "__main__":
    from src.prompts import get_clean_corrupted_pair
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    scores = ablate_all_heads(model, example)
    baseline = get_answer_logit_diff(model, model(example["prompt"]), example["correct"], example["incorrect"])
    print(f"unablated baseline: {baseline:.3f}")

    worst5 = sorted(scores, key=lambda x: x[2])[:5]  # most damaging when removed
    print("top 5 most important heads (largest drop when ablated):")
    for layer, head, diff in worst5:
        print(f"  L{layer}H{head}: ablated diff {diff:.3f} (drop of {baseline - diff:.3f})")