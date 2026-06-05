from src.model import load_model

def print_config(model):
    cfg = model.cfg
    print(f"n_layers: {cfg.n_layers}")
    print(f"n_heads: {cfg.n_heads}")
    print(f"d_model: {cfg.d_model}")
    print(f"d_head: {cfg.d_head}")
    print(f"n_ctx: {cfg.n_ctx}")

if __name__ == "__main__":
    model = load_model()
    print_config(model)