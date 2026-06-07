from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.patching import patch_activation

def get_answer_logit_diff(model, logits, correct_token: str, incorrect_token: str):
    correct_id = model.to_single_token(correct_token)
    incorrect_id = model.to_single_token(incorrect_token)
    last_logits = logits[0, -1]
    return (last_logits[correct_id] - last_logits[incorrect_id]).item()

if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()

    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    corrupted_logits = model(corrupted_tokens)

    patched_logits = patch_activation(model, corrupted_tokens, clean_cache, "resid_post", layer=0)

    clean_diff = get_answer_logit_diff(model, clean_logits, " Paris", " London")
    corrupted_diff = get_answer_logit_diff(model, corrupted_logits, " Paris", " London")
    patched_diff = get_answer_logit_diff(model, patched_logits, " Paris", " London")

    print(f"clean logit diff:     {clean_diff:.3f}")
    print(f"corrupted logit diff: {corrupted_diff:.3f}")
    print(f"patched logit diff:   {patched_diff:.3f}")