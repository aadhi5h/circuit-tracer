from src.model import load_model
from src.utils import get_answer_logit_diff
from src.agreement_dataset import build_agreement_dataset
from src.agreement_corruption import corrupt_example
from src.runner import run_layer_sweep

if __name__ == "__main__":
    model = load_model()
    examples = build_agreement_dataset()
    example = examples[0]  # "The key near the cabinets"
    corrupted = corrupt_example(example)

    clean_logits = model(example["prompt"])
    corrupted_logits = model(corrupted["prompt"])
    clean_diff = get_answer_logit_diff(model, clean_logits, example["correct"], example["incorrect"])
    corrupted_diff = get_answer_logit_diff(model, corrupted_logits, example["correct"], example["incorrect"])

    print(f"clean: {example['prompt']!r} -> diff {clean_diff:.3f}")
    print(f"corrupted: {corrupted['prompt']!r} -> diff {corrupted_diff:.3f}")

    layer_results = run_layer_sweep(model, example, corrupted["prompt"])
    print("layer sweep:")
    for layer, score in layer_results:
        print(f"  layer {layer}: {score:.3f}")