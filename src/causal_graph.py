from src.model import load_model
from src.prompts import get_clean_corrupted_pair
from src.utils import get_answer_logit_diff
from src.mlp_and_heads import patch_combo

class CausalGraph:
    def __init__(self):
        self.nodes = {}   # name -> {"type": "head"/"mlp", "layer": int, "head": int|None, "solo_score": float}
        self.edges = []   # (name_a, name_b, combined_score, additive_expectation)

    def add_node(self, name: str, node_type: str, layer: int, head: int = None, solo_score: float = None):
        self.nodes[name] = {"type": node_type, "layer": layer, "head": head, "solo_score": solo_score}

    def add_edge(self, name_a: str, name_b: str, combined_score: float, baseline: float):
        node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        gain_a = node_a["solo_score"] - baseline
        gain_b = node_b["solo_score"] - baseline
        combined_gain = combined_score - baseline
        additive_expectation = gain_a + gain_b
        self.edges.append({
            "a": name_a, "b": name_b,
            "combined": combined_score,
            "combined_gain": combined_gain,
            "additive_expectation": additive_expectation,
            "superadditive": combined_gain > additive_expectation,
        })

    def print_summary(self):
        print("nodes:")
        for name, data in self.nodes.items():
            print(f"  {name}: {data['type']} solo_score={data['solo_score']:.3f}")
        print("edges:")
        for e in self.edges:
            relation = "super-additive" if e["superadditive"] else "sub-additive"
            print(f"  {e['a']} + {e['b']}: combined={e['combined']:.3f} "
                  f"(naive sum would be {e['additive_expectation']:.3f}) -> {relation}")


if __name__ == "__main__":
    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    _, clean_cache = model.run_with_cache(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    def score(heads, mlps):
        logits = patch_combo(model, corrupted_tokens, clean_cache, heads, mlps)
        return get_answer_logit_diff(model, logits, " Paris", " London")

    graph = CausalGraph()
    graph.add_node("L8H11", "head", layer=8, head=11, solo_score=score([(8, 11)], []))
    graph.add_node("L10H0", "head", layer=10, head=0, solo_score=score([(10, 0)], []))
    graph.add_node("MLP0", "mlp", layer=0, solo_score=score([], [0]))

    graph.add_edge("L8H11", "L10H0", combined_score=score([(8, 11), (10, 0)], []), baseline=-3.470)
    graph.add_edge("L8H11", "MLP0", combined_score=score([(8, 11)], [0]), baseline=-3.470)
    graph.add_edge("L10H0", "MLP0", combined_score=score([(10, 0)], [0]), baseline=-3.470)

    graph.print_summary()