from src.model import load_model
from src.utils import get_answer_logit_diff
from src.targeted_patching import patch_at_position

CHECKPOINTS = [
    ("embed only", "embed", None),
    ("resid_pre layer 0", "resid_pre", 0),
    ("resid_mid layer 0", "resid_mid", 0),
    ("mlp_out layer 0", "mlp_out", 0),
    ("resid_post layer 6", "resid_post", 6),
    ("resid_post layer 8", "resid_post", 8),
    ("resid_post layer 10", "resid_post", 10),
]

def sweep_checkpoints(model, clean_prompt: str, corrupted_prompt: str, correct: str, incorrect: str):
    _, clean_cache = model.run_with_cache(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    last_pos = corrupted_tokens.shape[1] - 1

    clean_diff = get_answer_logit_diff(model, model(clean_prompt), correct, incorrect)
    corrupted_diff = get_answer_logit_diff(model, model(corrupted_prompt), correct, incorrect)

    results = {"clean": clean_diff, "corrupted": corrupted_diff, "checkpoints": []}
    for label, hook_name, layer in CHECKPOINTS:
        target_layer = layer if layer is not None else 0
        logits = patch_at_position(model, corrupted_tokens, clean_cache, hook_name, target_layer, last_pos)
        diff = get_answer_logit_diff(model, logits, correct, incorrect)
        recovery = (diff - corrupted_diff) / (clean_diff - corrupted_diff)
        results["checkpoints"].append({"label": label, "score": diff, "recovery": recovery})
    return results

def print_sweep(name: str, results: dict):
    print(f"=== {name} ===")
    print(f"clean: {results['clean']:.3f}  corrupted: {results['corrupted']:.3f}")
    for cp in results["checkpoints"]:
        print(f"  {cp['label']}: {cp['score']:.3f} ({cp['recovery']:.1%} recovery)")

if __name__ == "__main__":
    from src.prompts import get_clean_corrupted_pair
    from src.ioi_dataset import build_ioi_dataset
    from src.ioi_corruption import corrupt_example

    model = load_model()

    clean, corrupted = get_clean_corrupted_pair()
    pl_results = sweep_checkpoints(model, clean, corrupted, " Paris", " London")
    print_sweep("Paris/London", pl_results)

    ioi_example = build_ioi_dataset()[0]
    ioi_corrupted = corrupt_example(ioi_example)
    ioi_results = sweep_checkpoints(model, ioi_example["prompt"], ioi_corrupted["prompt"],
                                      ioi_example["correct"], ioi_example["incorrect"])
    print_sweep("IOI", ioi_results)