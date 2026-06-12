from src.model import load_model
from src.automated_comparison import run_comparison

def sweep_all_layers(model, correct_token: str = " Paris", incorrect_token: str = " London"):
    results = []
    for layer in range(model.cfg.n_layers):
        result = run_comparison(model, layer, correct_token, incorrect_token)
        result["layer"] = layer
        results.append(result)
    return results

if __name__ == "__main__":
    model = load_model()
    results = sweep_all_layers(model)
    for r in results:
        print(f"layer {r['layer']}: clean={r['clean']:.3f} corrupted={r['corrupted']:.3f} patched={r['patched']:.3f}")