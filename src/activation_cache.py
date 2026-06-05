from src.model import load_model

def get_cache(model, prompt: str):
    logits, cache = model.run_with_cache(prompt)
    return logits, cache

if __name__ == "__main__":
    model = load_model()
    logits, cache = get_cache(model, "The quick brown fox")
    print(list(cache.keys())[:5])