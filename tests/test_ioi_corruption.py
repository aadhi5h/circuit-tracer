from src.model import load_model
from src.utils import get_answer_logit_diff
from src.ioi_dataset import build_ioi_dataset
from src.ioi_corruption import corrupt_example

def test_corruption_reduces_logit_diff():
    model = load_model()
    examples = build_ioi_dataset()[:20]  # subset for speed

    clean_diffs = []
    corrupted_diffs = []
    for ex in examples:
        clean_logits = model(ex["prompt"])
        clean_diff = get_answer_logit_diff(model, clean_logits, ex["correct"], ex["incorrect"])
        clean_diffs.append(clean_diff)

        corrupted = corrupt_example(ex)
        corrupted_logits = model(corrupted["prompt"])
        corrupted_diff = get_answer_logit_diff(model, corrupted_logits, ex["correct"], ex["incorrect"])
        corrupted_diffs.append(corrupted_diff)

    clean_mean = sum(clean_diffs) / len(clean_diffs)
    corrupted_mean = sum(corrupted_diffs) / len(corrupted_diffs)

    assert corrupted_mean < clean_mean, (
        f"expected corruption to reduce logit diff, got clean={clean_mean:.3f} "
        f"corrupted={corrupted_mean:.3f}"
    )
    print(f"clean mean: {clean_mean:.3f}, corrupted mean: {corrupted_mean:.3f}")

if __name__ == "__main__":
    test_corruption_reduces_logit_diff()