from src.model import load_model

def trace_residual_contributions(model, prompt: str):
    _, cache = model.run_with_cache(prompt)
    decomposed, labels = cache.decompose_resid(return_labels=True)
    # decomposed shape: [n_components, batch, seq, d_model]
    return decomposed, labels

if __name__ == "__main__":
    model = load_model()
    prompt = "The Eiffel Tower is located in the city of Paris"
    decomposed, labels = trace_residual_contributions(model, prompt)
    norms = decomposed.norm(dim=-1)[:, 0, -1]  # norm at last token, per component
    for label, norm in zip(labels, norms):
        print(f"{label}: {norm.item():.3f}")