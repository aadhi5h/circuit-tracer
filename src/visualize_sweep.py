from src.model import load_model
from src.layer_sweep import sweep_all_layers

def print_bar_chart(results, width: int = 40):
    max_patched = max(r["patched"] for r in results)
    min_patched = min(r["patched"] for r in results)
    span = max_patched - min_patched or 1

    for r in results:
        normalized = (r["patched"] - min_patched) / span
        bar_len = int(normalized * width)
        bar = "#" * bar_len
        print(f"layer {r['layer']:2d} | {bar} {r['patched']:.3f}")

if __name__ == "__main__":
    model = load_model()
    results = sweep_all_layers(model)
    print_bar_chart(results)