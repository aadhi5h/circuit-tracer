from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    model_name: str = "gpt2"
    clean_prompt: str = "The Eiffel Tower is located in the city of Paris"
    corrupted_prompt: str = "The Eiffel Tower is located in the city of London"
    correct_token: str = " Paris"
    incorrect_token: str = " London"
    seed: int = 0

DEFAULT_CONFIG = ExperimentConfig()

if __name__ == "__main__":
    print(DEFAULT_CONFIG)