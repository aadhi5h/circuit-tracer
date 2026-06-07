from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.patching import patch_activation

def test_patch_shapes():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()

    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    n_tokens = corrupted_tokens.shape[1]

    logits = patch_activation(model, corrupted_tokens, clean_cache, "resid_post", layer=0)

    assert logits.shape == (1, n_tokens, model.cfg.d_vocab)
    print("patching shape check passed")

if __name__ == "__main__":
    test_patch_shapes()