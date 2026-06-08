from src.model import load_model
from src.induction import run_induction_experiment

def get_induction_head_scores(model, cache, seq_len: int):
    scores = {}
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer][0]  # [n_heads, seq, seq]
        n_heads = pattern.shape[0]
        for head in range(n_heads):
            offset_attn = pattern[head].diagonal(offset=-(seq_len - 1))
            if offset_attn.numel() > 0:
                scores[(layer, head)] = offset_attn.mean().item()
    return scores

if __name__ == "__main__":
    model = load_model()
    seq_len = 20
    tokens, logits, cache = run_induction_experiment(model, seq_len)
    scores = get_induction_head_scores(model, cache, seq_len)
    top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
    for (layer, head), score in top5:
        print(f"layer {layer} head {head}: induction score {score:.3f}")