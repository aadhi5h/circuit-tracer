from collections import defaultdict
from src.model import load_model
from src.ioi_dataset import build_ioi_dataset
from src.ioi_head_scan import scan_heads_averaged

def group_by_layer(ranked_heads):
    by_layer = defaultdict(list)
    for (layer, head), score in ranked_heads:
        by_layer[layer].append(score)
    return {layer: max(scores) for layer, scores in by_layer.items()}  # max instead of mean

def print_layer_bar_chart(layer_means: dict, width: int = 40):
    max_score = max(layer_means.values())
    min_score = min(layer_means.values())
    span = max_score - min_score or 1

    for layer in sorted(layer_means):
        score = layer_means[layer]
        normalized = (score - min_score) / span
        bar_len = int(normalized * width)
        bar = "#" * bar_len
        print(f"layer {layer:2d} | {bar} {score:.3f}")

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()[:10]  # smaller subset for speed, this reruns full 144-head scan
    ranked = scan_heads_averaged(model, examples)
    layer_means = group_by_layer(ranked)
    print_layer_bar_chart(layer_means)