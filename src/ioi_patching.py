from src.model import load_model
from src.utils import get_answer_logit_diff
from src.patching import patch_activation
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example

def patch_ioi_layer(model, example: dict, layer: int):
    clean_prompt = example["prompt"]
    corrupted = corrupt_example(example)

    _, clean_cache = model.run_with_cache(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted["prompt"])

    clean_logits = model(clean_prompt)
    corrupted_logits = model(corrupted_tokens)
    patched_logits = patch_activation(model, corrupted_tokens, clean_cache, "resid_post", layer=layer)

    return {
        "clean": get_answer_logit_diff(model, clean_logits, example["correct"], example["incorrect"]),
        "corrupted": get_answer_logit_diff(model, corrupted_logits, example["correct"], example["incorrect"]),
        "patched": get_answer_logit_diff(model, patched_logits, example["correct"], example["incorrect"]),
    }

if __name__ == "__main__":
    model = load_model()
    examples = build_ioi_dataset()
    example = examples[0]  # single example first, sweep comes later

    for layer in range(model.cfg.n_layers):
        result = patch_ioi_layer(model, example, layer)
        print(f"layer {layer}: {result}")