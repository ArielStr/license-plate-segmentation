from training.configs.experiment_config import ExperimentConfig
from training.configs.experiments_v1 import EXPERIMENTS_V1
from training.configs.experiments_v2 import EXPERIMENTS_V2
from training.configs.experiments_v3 import EXPERIMENTS_V3


EXPERIMENTS = {
    **EXPERIMENTS_V1,
    **EXPERIMENTS_V2,
    **EXPERIMENTS_V3,
}


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available experiments: {available}"
        )

    return EXPERIMENTS[name]