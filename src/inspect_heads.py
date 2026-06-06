from src.model import load_model
from src.activation_cache import get_cache

def get_attention_pattern(cache, layer: int):
    return cache["pattern", layer][0]  # drop batch dim -> [n_heads, seq_len, seq_len]

if __name__ == "__main__":
    model = load_model()
    _, cache = get_cache(model, "The quick brown fox")
    pattern = get_attention_pattern(cache, layer=0)
    print(f"layer 0 pattern shape: {pattern.shape}")
    print(f"head 0 attention (first row): {pattern[0, -1]}")