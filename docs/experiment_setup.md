# Reproducible Experiment Setup

## Config
`src/config.py` defines `ExperimentConfig`, a dataclass holding the model name,
clean/corrupted prompt pair, target tokens, and seed for a given experiment.
`DEFAULT_CONFIG` is the Eiffel Tower / Paris-London pair used in Days 5-11.

## Why
Every script through Day 11 hardcoded `" Paris"` / `" London"` inline. Centralizing
these in one config means swapping to a new clean/corrupted pair (planned for
the independent circuit investigation later in the project) only requires
changing one file instead of every script.

## Status
[Note: pick one]
- Scripts through Day 11 have been updated to import from `DEFAULT_CONFIG`.
- Scripts through Day 11 still use hardcoded strings; migration deferred —
  new scripts from Day 12 onward should use `ExperimentConfig` directly.

## Usage
```python
from src.config import DEFAULT_CONFIG

clean = DEFAULT_CONFIG.clean_prompt
correct = DEFAULT_CONFIG.correct_token
```