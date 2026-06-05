from transformer_lens import HookedTransformer

def load_model(name: str = "gpt2") -> HookedTransformer:
    return HookedTransformer.from_pretrained(name, device="cpu")

if __name__ == "__main__":
    model = load_model()
    print(model.cfg.model_name)