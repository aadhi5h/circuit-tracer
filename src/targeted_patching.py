from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff, full_hook_name

def patch_at_position(model, corrupted_tokens, clean_cache, hook_name: str, layer: int, position: int):
    def hook_fn(activation, hook):
        activation[:, position, :] = clean_cache[hook_name, layer][:, position, :]
        return activation
    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(full_hook_name(layer, hook_name), hook_fn)]
    )
    return logits

def scan_positions(model, corrupted_tokens, clean_cache, hook_name: str, layer: int, correct: str, incorrect: str):
    n_pos = corrupted_tokens.shape[1]
    results = []
    for pos in range(n_pos):
        logits = patch_at_position(model, corrupted_tokens, clean_cache, hook_name, layer, pos)
        diff = get_answer_logit_diff(model, logits, correct, incorrect)
        results.append((pos, diff))
    return results

if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    tokens_str = model.to_str_tokens(corrupted)

    print("scanning positions for resid_mid (layer 0):")
    results_in = scan_positions(model, corrupted_tokens, clean_cache, "resid_mid", 0, " Paris", " London")
    for pos, diff in results_in:
        print(f"  pos {pos} ({tokens_str[pos]!r}): {diff:.3f}")

    print("scanning positions for mlp_out (layer 0):")
    results_out = scan_positions(model, corrupted_tokens, clean_cache, "mlp_out", 0, " Paris", " London")
    for pos, diff in results_out:
        print(f"  pos {pos} ({tokens_str[pos]!r}): {diff:.3f}")