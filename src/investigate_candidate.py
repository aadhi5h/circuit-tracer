from src.model import load_model
from src.utils import get_answer_logit_diff
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example
from src.ioi_head_scan import patch_single_head_ioi

def patch_reverse(model, clean_tokens, corrupted_cache, layer: int, head: int):
    """Patch corrupted activation INTO clean run - reverse direction."""
    def hook_fn(activation, hook):
        activation[:, -1, head, :] = corrupted_cache["z", layer][:, -1, head, :]
        return activation

    logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]
    )
    return logits

def investigate_head(model, examples: list, layer: int, head: int):
    forward_scores = []   # corrupted->clean patch (original direction)
    reverse_scores = []   # clean->corrupted patch (reverse direction)

    for ex in examples:
        corrupted = corrupt_example(ex)
        clean_tokens = model.to_tokens(ex["prompt"])
        corrupted_tokens = model.to_tokens(corrupted["prompt"])

        _, clean_cache = model.run_with_cache(ex["prompt"])
        _, corrupted_cache = model.run_with_cache(corrupted["prompt"])

        fwd_logits = patch_single_head_ioi(model, corrupted_tokens, clean_cache, layer, head)
        fwd_diff = get_answer_logit_diff(model, fwd_logits, ex["correct"], ex["incorrect"])
        forward_scores.append(fwd_diff)

        rev_logits = patch_reverse(model, clean_tokens, corrupted_cache, layer, head)
        rev_diff = get_answer_logit_diff(model, rev_logits, ex["correct"], ex["incorrect"])
        reverse_scores.append(rev_diff)

    return {
        "forward_mean": sum(forward_scores) / len(forward_scores),
        "reverse_mean": sum(reverse_scores) / len(reverse_scores),
    }

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()[:10]

    candidates = [(10, 0), (9, 9), (8, 6)]  # L10H0 (the puzzle), plus top-2 for comparison
    for layer, head in candidates:
        result = investigate_head(model, examples, layer, head)
        print(f"L{layer}H{head}: forward={result['forward_mean']:.3f} reverse={result['reverse_mean']:.3f}")