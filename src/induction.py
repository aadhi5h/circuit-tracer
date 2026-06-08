import torch
from src.model import load_model

def make_repeated_tokens(model, seq_len: int = 20, batch: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    random_tokens = torch.randint(1000, 10000, (batch, seq_len))
    repeated = torch.cat([random_tokens, random_tokens], dim=1)
    return repeated

def run_induction_experiment(model, seq_len: int = 20):
    tokens = make_repeated_tokens(model, seq_len)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache

if __name__ == "__main__":
    model = load_model()
    tokens, logits, cache = run_induction_experiment(model)
    preds = logits.argmax(dim=-1)
    # compare predictions on second half against actual repeated tokens
    second_half_actual = tokens[0, 21:]
    second_half_preds = preds[0, 20:-1]
    accuracy = (second_half_actual == second_half_preds).float().mean().item()
    print(f"induction accuracy on repeated sequence: {accuracy:.3f}")