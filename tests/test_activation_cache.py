from src.model import load_model
from src.activation_cache import get_cache

def test_cache_shapes():
    model = load_model()
    prompt = "The quick brown fox"
    logits, cache = get_cache(model, prompt)

    n_tokens = model.to_tokens(prompt).shape[1]

    assert logits.shape == (1, n_tokens, model.cfg.d_vocab)
    assert cache["pattern", 0].shape == (1, model.cfg.n_heads, n_tokens, n_tokens)
    assert cache["resid_post", 0].shape == (1, n_tokens, model.cfg.d_model)
    print("all shape checks passed")

if __name__ == "__main__":
    test_cache_shapes()