from src.model import load_model
from src.utils import get_answer_logit_diff
from src.ioi_dataset import build_ioi_dataset

def test_ioi_baseline_behavior():
    model = load_model()
    examples = build_ioi_dataset()[:20]  # subset for speed

    diffs = []
    for ex in examples:
        logits = model(ex["prompt"])
        diff = get_answer_logit_diff(model, logits, ex["correct"], ex["incorrect"])
        diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    assert avg_diff > 0, f"expected positive average logit diff for IOI task, got {avg_diff:.3f}"
    print(f"IOI baseline avg logit diff: {avg_diff:.3f} over {len(diffs)} examples")

if __name__ == "__main__":
    test_ioi_baseline_behavior()