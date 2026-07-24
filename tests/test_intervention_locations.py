from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.targeted_patching import patch_at_position

def test_input_vs_output_patching():
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)
    last_pos = corrupted_tokens.shape[1] - 1

    in_logits = patch_at_position(model, corrupted_tokens, clean_cache, "resid_mid", 0, last_pos)
    in_diff = get_answer_logit_diff(model, in_logits, " Paris", " London")

    out_logits = patch_at_position(model, corrupted_tokens, clean_cache, "mlp_out", 0, last_pos)
    out_diff = get_answer_logit_diff(model, out_logits, " Paris", " London")

    print(f"patch resid_mid (last pos): {in_diff:.3f}")
    print(f"patch mlp_out (last pos): {out_diff:.3f}")

    assert in_diff > -3.470
    assert out_diff > -3.470
    # resid_mid patching (pre-MLP0) should recover more than mlp_out alone,
    # since it's closer to full clean and captures embed+pos+attn0 too
    assert in_diff > out_diff

if __name__ == "__main__":
    test_input_vs_output_patching()