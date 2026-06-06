from src.model import load_model
from src.activation_cache import get_cache

def token_residuals(model, cache, prompt: str):
    tokens = model.to_str_tokens(prompt)
    resid = cache["resid_post", -1]  # last layer, shape [1, seq, d_model]
    return tokens, resid

if __name__ == "__main__":
    model = load_model()
    prompt = "The quick brown fox"
    _, cache = get_cache(model, prompt)
    tokens, resid = token_residuals(model, cache, prompt)
    for i, tok in enumerate(tokens):
        print(f"{i}: {tok!r} -> norm {resid[0, i].norm().item():.3f}")