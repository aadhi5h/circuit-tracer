from src.model import load_model
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example
from src.ioi_head_scan import scan_heads_averaged
from src.sweep_runner import sweep_checkpoints

def build_final_graph(model, n_examples: int = 10):
    examples = build_ioi_dataset()[:n_examples]

    # top heads from averaged scan
    ranked_heads = scan_heads_averaged(model, examples)
    top_heads = ranked_heads[:5]

    # layer-level recovery curve (already validated robust across templates in Day 56)
    ex = examples[0]
    corrupted = corrupt_example(ex)
    sweep = sweep_checkpoints(model, ex["prompt"], corrupted["prompt"], ex["correct"], ex["incorrect"])

    return {
        "top_heads": top_heads,
        "layer_sweep": sweep["checkpoints"],
        "clean_baseline": sweep["clean"],
        "corrupted_baseline": sweep["corrupted"],
    }

def print_graph(graph: dict):
    print("=== Final IOI Attribution Graph ===")
    print(f"clean: {graph['clean_baseline']:.3f}  corrupted: {graph['corrupted_baseline']:.3f}")
    print()
    print("top heads (averaged patching score):")
    for (layer, head), score in graph["top_heads"]:
        print(f"  L{layer}H{head}: {score:.3f}")
    print()
    print("layer-level recovery:")
    for cp in graph["layer_sweep"]:
        print(f"  {cp['label']}: {cp['recovery']:.1%}")

if __name__ == "__main__":
    model = load_model()
    graph = build_final_graph(model, n_examples=10)
    print_graph(graph)