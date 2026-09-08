from dataclasses import dataclass, fields
from typing import Literal


class BaseConfig:
    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class EnvSetup(BaseConfig):
    # id: str = "ALE/Frogger-v5"
    id: str = "ALE/Breakout-v5"
    # id: str = "ALE/SpaceInvaders-v5"
    mode: Literal[0, 1, 2] = 0
    difficulty: Literal[0, 1] = 0


AUTO_FIRE: bool = True
MAX_EPISODE_STEPS: int = 1000


@dataclass
class TrainingConfig(BaseConfig):
    env_id: str = EnvSetup.id
    seed: int = 1000
    num_envs: int = 8
    num_steps: int = 128
    total_timesteps: int = 10_000_000
    total_updates: int = total_timesteps // (num_envs * num_steps)
    checkpoint_window: int = 20
    learning_rate: float = 2.5e-4
    resume_checkpoint: str | None = None

    @property
    def run_name(self) -> str:
        """Nombre descriptivo para comparar corridas en TensorBoard/checkpoints.
        Se arma a partir de env_id, asi que cambiar de juego genera
        automaticamente un run_name distinto sin tocar nada mas."""
        env_slug = self.env_id.split("/")[-1]
        return (
            f"{env_slug}"
            f"__envs_{self.num_envs}"
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
class AtariPreprocessingArgs(BaseConfig):
    noop_max: int = 50
    frame_skip: int = 4
    screen_size: int = 84
    terminal_on_life_loss: bool = False
    grayscale_obs: bool = True
    scale_obs: bool = False
