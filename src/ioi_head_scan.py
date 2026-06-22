from src.model import load_model
from src.utils import get_answer_logit_diff
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example

def patch_single_head_ioi(model, corrupted_tokens, clean_cache, layer: int, head: int):
    def hook_fn(activation, hook):
        activation[:, -1, head, :] = clean_cache["z", layer][:, -1, head, :]
        return activation

    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]
    )
    return logits

def scan_heads_on_example(model, example: dict):
    clean_prompt = example["prompt"]
    corrupted = corrupt_example(example)

    _, clean_cache = model.run_with_cache(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted["prompt"])

    scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            logits = patch_single_head_ioi(model, corrupted_tokens, clean_cache, layer, head)
            diff = get_answer_logit_diff(model, logits, example["correct"], example["incorrect"])
            scores.append((layer, head, diff))
    return scores

def scan_heads_averaged(model, examples: list):
    """Average head scores across multiple examples for a more robust ranking."""
    n = len(examples)
    accum = {}
    for ex in examples:
        scores = scan_heads_on_example(model, ex)
        for layer, head, diff in scores:
            accum[(layer, head)] = accum.get((layer, head), 0) + diff / n
    return sorted(accum.items(), key=lambda x: -x[1])

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()[:5]  # small subset — full scan is 144 patches PER example
    ranked = scan_heads_averaged(model, examples)
    print("top 5 heads (averaged over 5 examples):")
    for (layer, head), score in ranked[:5]:
        print(f"  L{layer}H{head}: {score:.3f}")