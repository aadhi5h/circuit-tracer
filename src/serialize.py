import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

RESULTS_DIR = "experiments/results"

@dataclass
class ExperimentResult:
    name: str
    hook_name: str
    layer_results: list = field(default_factory=list)   # [(layer, score), ...]
    head_results: list = field(default_factory=list)     # [(layer, head, score), ...]
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return asdict(self)

    def save(self) -> str:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        safe_ts = self.timestamp.replace(":", "-")
        path = os.path.join(RESULTS_DIR, f"{self.name}_{safe_ts}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"saved result to {path}")
        return path

    @staticmethod
    def load(path: str) -> "ExperimentResult":
        with open(path) as f:
            data = json.load(f)
        return ExperimentResult(**data)


if __name__ == "__main__":
    from src.model import load_model
    from src.prompts import get_clean_corrupted_pair
    from src.runner import run_layer_sweep, run_head_sweep

    model = load_model()
    clean, corrupted = get_clean_corrupted_pair()
    example = {"prompt": clean, "correct": " Paris", "incorrect": " London"}

    layer_results = run_layer_sweep(model, example, corrupted)
    head_results = run_head_sweep(model, example, corrupted)

    result = ExperimentResult(
        name="paris_london_sweep",
        hook_name="resid_post",
        layer_results=layer_results,
        head_results=head_results,
        metadata={"clean_prompt": clean, "corrupted_prompt": corrupted},
    )
    path = result.save()

    loaded = ExperimentResult.load(path)
    print(f"round-trip check: {len(loaded.head_results)} head results loaded back")