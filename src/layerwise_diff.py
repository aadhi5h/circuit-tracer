from src.model import load_model
from src.prompts import get_clean_corrupted_pair

def layerwise_resid_diff(model, clean: str, corrupted: str):
    _, clean_cache = model.run_with_cache(clean)
    _, corrupted_cache = model.run_with_cache(corrupted)

    diffs = []
    for layer in range(model.cfg.n_layers):
        clean_resid = clean_cache["resid_post", layer][0, -1]
        corrupted_resid = corrupted_cache["resid_post", layer][0, -1]
        diff = (clean_resid - corrupted_resid).norm().item()
        diffs.append(diff)
    return diffs

if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    diffs = layerwise_resid_diff(model, clean, corrupted)
    for layer, diff in enumerate(diffs):
        print(f"layer {layer}: resid diff norm {diff:.3f}")