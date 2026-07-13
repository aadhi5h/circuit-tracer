from collections import defaultdict
from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.ablation import ablate_all_heads

def group_by_layer_min(scores):
    """Use min (most damaging) per layer, not mean - same reasoning as Day 26's max fix."""
    by_layer = defaultdict(list)
    for layer, head, diff in scores:
        by_layer[layer].append(diff)
    return {layer: min(diffs) for layer, diffs in by_layer.items()}

def print_bar_chart(layer_scores: dict, width: int = 40):
    max_score = max(layer_scores.values())
    min_score = min(layer_scores.values())
    span = max_score - min_score or 1
    for layer in sorted(layer_scores):
        score = layer_scores[layer]
        normalized = (score - min_score) / span
        bar_len = int(normalized * width)
        print(f"layer {layer:2d} | {'#' * bar_len} {score:.3f}")

if __name__ == "__main__":
    model = load_model()
    clean, _ = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    scores = ablate_all_heads(model, example)
    layer_scores = group_by_layer_min(scores)
    print_bar_chart(layer_scores)