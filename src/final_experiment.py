from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.runner import run_layer_sweep, run_head_sweep
from src.serialize import ExperimentResult

def run_final_experiment():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    layer_results = run_layer_sweep(model, example, corrupted)
    head_results = run_head_sweep(model, example, corrupted)

    top_head = max(head_results, key=lambda x: x[2])
    top_layer = max(layer_results, key=lambda x: x[1])

    result = ExperimentResult(
        name="paris_london_final",
        hook_name="resid_post",
        layer_results=layer_results,
        head_results=head_results,
        metadata={
            "clean_prompt": clean,
            "corrupted_prompt": corrupted,
            "top_head": f"L{top_head[0]}H{top_head[1]}",
            "top_head_score": top_head[2],
            "top_layer": top_layer[0],
            "top_layer_score": top_layer[1],
        },
    )
    path = result.save()
    return result, path

if __name__ == "__main__":
    result, path = run_final_experiment()
    print(f"finalized experiment saved to {path}")
    print(f"top head: {result.metadata['top_head']} ({result.metadata['top_head_score']:.3f})")
    print(f"top layer: {result.metadata['top_layer']} ({result.metadata['top_layer_score']:.3f})")