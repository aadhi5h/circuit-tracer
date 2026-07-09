from src.model import load_model
from src.utils import get_answer_logit_diff
from src.agreement_dataset import build_agreement_dataset

def test_agreement_baseline_behavior():
    model = load_model()
    examples = build_agreement_dataset()

    diffs = []
    for ex in examples:
        logits = model(ex["prompt"])
        diff = get_answer_logit_diff(model, logits, ex["correct"], ex["incorrect"])
        diffs.append({"prompt": ex["prompt"], "diff": diff})

    scores = [d["diff"] for d in diffs]
    avg_diff = sum(scores) / len(scores)
    n_correct = sum(1 for s in scores if s > 0)

    print(f"agreement baseline avg logit diff: {avg_diff:.3f} over {len(scores)} examples")
    print(f"accuracy (diff > 0): {n_correct}/{len(scores)} = {n_correct/len(scores):.3f}")

    worst3 = sorted(diffs, key=lambda d: d["diff"])[:3]
    print("worst 3 examples:")
    for d in worst3:
        print(f"  {d['prompt']!r} -> diff {d['diff']:.3f}")

    assert avg_diff > 0, f"expected positive average logit diff for agreement task, got {avg_diff:.3f}"

if __name__ == "__main__":
    test_agreement_baseline_behavior()