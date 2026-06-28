import json
import os
from datetime import datetime, timezone

LOG_DIR = "experiments/logs"

def log_result(name: str, data: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(LOG_DIR, f"{name}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"logged to {path}")
    return path

if __name__ == "__main__":
    log_result("test_log", {"example": "value", "score": 1.234})