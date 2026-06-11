def get_answer_logit_diff(model, logits, correct_token: str, incorrect_token: str):
    correct_id = model.to_single_token(correct_token)
    incorrect_id = model.to_single_token(incorrect_token)
    last_logits = logits[0, -1]
    return (last_logits[correct_id] - last_logits[incorrect_id]).item()


def full_hook_name(layer: int, hook_name: str) -> str:
    return f"blocks.{layer}.hook_{hook_name}"