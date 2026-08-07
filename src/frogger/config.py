from dataclasses import dataclass, fields
from typing import Literal


class BaseConfig:
    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class TrainingConfig(BaseConfig):
    seed: int = 2000
    num_envs: int = 8
    num_steps: int = 128
    total_timesteps: int = 5_000_000
    total_updates: int = total_timesteps // (num_envs * num_steps)
    checkpoint_window: int = 20
    learning_rate: float = 2.5e-4
    resume_checkpoint: str | None = None
    # resume_checkpoint: str | None = (
    #     r"checkpoints\Frogger__envs_8__steps_128__lr_0.00025__seed_2000\best_agent_score_9.00.pt"
    # )

    @property
    def run_name(self) -> str:
        """Nombre descriptivo para comparar corridas automáticamente en TensorBoard."""
        return (
            f"Frogger__envs_{self.num_envs}"
            f"__steps_{self.num_steps}"
            f"__lr_{self.learning_rate}"
            f"__seed_{self.seed}"
        )


@dataclass
class GeneralConfig(BaseConfig):
    obs_type: Literal["rgb", "ram", "grayscale"] = "rgb"
    full_action_space: bool = False
    frameskip: int = 1
    render_mode: Literal["human", "rgb_array", "ansi", "rgb_array_list"] | None = None


@dataclass
class EnvSetup(BaseConfig):
    id: str = "ALE/Frogger-v5"
    mode: Literal[0, 1, 2] = 0
    difficulty: Literal[0, 1] = 0


@dataclass
class AtariPreprocessingArgs(BaseConfig):
    noop_max: int = 30
    frame_skip: int = 2
    screen_size: int = 84
    terminal_on_life_loss: bool = True
    grayscale_obs: bool = True
    scale_obs: bool = False
