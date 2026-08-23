import glob
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from deep_rl_agent.agent import AgentOutput, ImpalaPPOAgent
from deep_rl_agent.config import (
    AtariPreprocessingArgs,
    EnvSetup,
    GeneralConfig,
    TrainingConfig,
)

t_config = TrainingConfig()

os.makedirs("logs", exist_ok=True)
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=rf"logs/{t_config.run_name}.log",
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class TrainingState:
    global_step: int = 0
    start_update: int = 1
    best_avg_return: float = -float("inf")
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=t_config.checkpoint_window))
    recent_returns_wide: deque = field(default_factory=lambda: deque(maxlen=200))


def get_next_run_name(base_run_name: str, runs_dir: str = "runs") -> str:
    pattern = os.path.join(runs_dir, f"{base_run_name}_resume_*")
    existing_resumes = glob.glob(pattern)

    if not existing_resumes:
        return f"{base_run_name}_resume_1"

    indices = []
    for path in existing_resumes:
        match = re.search(r"_resume_(\d+)$", path)
        if match:
            indices.append(int(match.group(1)))

    next_idx = max(indices, default=0) + 1
    return f"{base_run_name}_resume_{next_idx}"


def load_training_state(agent: ImpalaPPOAgent, config: TrainingConfig) -> TrainingState:
    state = TrainingState()

    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        logger.info(f"Cargando checkpoint desde: {config.resume_checkpoint}")

        checkpoint_data: dict = agent.load(config.resume_checkpoint)

        state.global_step = checkpoint_data.get("global_step", 0)
        state.best_avg_return = checkpoint_data.get("best_avg_return", -float("inf"))

        saved_returns = checkpoint_data.get("recent_returns", [])
        state.recent_returns.extend(saved_returns)
        state.recent_returns_wide.extend(saved_returns)

        state.start_update = (state.global_step // (config.num_envs * config.num_steps)) + 1

        logger.info(
            f"Reanudando en el paso global {state.global_step} (Update {state.start_update})"
        )

    return state


def create_rollout_buffers(
    obs_shape: tuple[int, ...], num_steps: int, num_envs: int, device: torch.device
) -> dict[str, torch.Tensor]:
    buffer_dim: tuple[int, int] = (num_steps, num_envs)

    return {
        "obs": torch.zeros((*buffer_dim, *obs_shape), dtype=torch.uint8, device=device),
        "actions": torch.zeros(buffer_dim, dtype=torch.long, device=device),
        "log_probs": torch.zeros(buffer_dim, device=device),
        "rewards": torch.zeros(buffer_dim, device=device),
        "dones": torch.zeros(buffer_dim, device=device),
        "values": torch.zeros(buffer_dim, device=device),
    }


def make_env(seed_offset: int = 0):

    def thunk() -> gym.core.Env:
        gym.register_envs(ale_py)
        env = gym.make(**GeneralConfig(), **EnvSetup())
        env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingArgs())
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.action_space.seed(t_config.seed + seed_offset)
        return env

    return thunk


def log_finished_episodes(infos: dict, state: TrainingState, writer: SummaryWriter) -> None:
    if "episode" not in infos:
        return

    for idx, finished in enumerate(infos["_episode"]):
        if not finished:
            continue

        ep_return = float(infos["episode"]["r"][idx])
        ep_length = int(infos["episode"]["l"][idx])

        state.recent_returns.append(ep_return)
        state.recent_returns_wide.append(ep_return)

        writer.add_scalar("Eval/episodic_return", ep_return, state.global_step)
        writer.add_scalar("Eval/episodic_length", ep_length, state.global_step)
        writer.add_scalar(
            "Eval/max_return_recent_200", max(state.recent_returns_wide), state.global_step
        )


def collect_rollout(
    envs: gym.vector.VectorEnv,
    agent: ImpalaPPOAgent,
    buffers: dict[str, torch.Tensor],
    obs: np.ndarray,
    state: TrainingState,
    writer: SummaryWriter,
    device: torch.device,
) -> np.ndarray:
    num_steps = buffers["actions"].shape[0]

    for step in range(num_steps):
        state.global_step += envs.num_envs
        obs_tensor = torch.tensor(obs, device=device)
        buffers["obs"][step] = obs_tensor

        with torch.no_grad():
            output: AgentOutput = agent.get_action_and_value(obs_tensor)

        buffers["actions"][step] = output.actions
        buffers["log_probs"][step] = output.log_probs
        buffers["values"][step] = output.values.flatten()

        next_obs, rewards, terminations, truncations, infos = envs.step(
            output.actions.cpu().numpy()
        )
        buffers["rewards"][step] = torch.tensor(rewards, device=device)
        buffers["dones"][step] = torch.tensor(
            np.logical_or(terminations, truncations), device=device
        )

        log_finished_episodes(infos, state, writer)
        obs = next_obs

    return obs


def log_scalars(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def evaluate_and_save_checkpoint(
    agent: ImpalaPPOAgent,
    state: TrainingState,
    checkpoint_dir: str,
    writer: SummaryWriter,
    window_size: int,
) -> None:
    """Evalúa el promedio móvil y guarda un nuevo checkpoint si es récord."""
    if len(state.recent_returns) < window_size:
        return

    current_avg_return = float(np.mean(state.recent_returns))
    writer.add_scalar("Eval/avg_return_window", current_avg_return, state.global_step)

    if current_avg_return <= state.best_avg_return:
        return

    state.best_avg_return = current_avg_return
    ckpt_path = os.path.join(checkpoint_dir, f"best_agent_score_{state.best_avg_return:.2f}.pt")
    agent.save(ckpt_path, state.global_step, list(state.recent_returns), state.best_avg_return)
    logger.info(
        f"[Paso {state.global_step}] Nuevo récord ({state.best_avg_return:.2f}). Checkpoint guardado en {ckpt_path}"
    )


def main() -> None:
    random.seed(t_config.seed)
    np.random.seed(t_config.seed)
    torch.manual_seed(t_config.seed)
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    checkpoint_dir = os.path.join("checkpoints", t_config.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_fns = [make_env(seed_offset=i) for i in range(t_config.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    agent = ImpalaPPOAgent(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        lr=t_config.learning_rate,
        update_epochs=4,
        num_minibatches=4,
    )

    state: TrainingState = load_training_state(agent, t_config)

    if t_config.resume_checkpoint and os.path.exists(t_config.resume_checkpoint):
        tensorboard_run_name = get_next_run_name(t_config.run_name)
    else:
        tensorboard_run_name = t_config.run_name

    writer = SummaryWriter(rf"runs/{tensorboard_run_name}")
    logger.info(
        f"Entrenando '{t_config.env_id}'. Logs de TensorBoard en: runs/{tensorboard_run_name}"
    )

    buffers: dict[str, torch.Tensor] = create_rollout_buffers(
        obs_shape=envs.single_observation_space.shape,
        num_steps=t_config.num_steps,
        num_envs=t_config.num_envs,
        device=device,
    )

    initial_global_step: int = state.global_step
    start_time: float = time.time()
    obs, info = envs.reset(seed=t_config.seed)

    for update in range(state.start_update, t_config.total_updates + 1):
        obs = collect_rollout(envs, agent, buffers, obs, state, writer, device)

        with torch.no_grad():
            next_value = agent.get_value(torch.tensor(obs, device=device))

        metrics = agent.update({**buffers, "next_value": next_value})
        log_scalars(writer, "losses", metrics, state.global_step)

        sps = int((state.global_step - initial_global_step) / (time.time() - start_time))
        writer.add_scalar("Performance/SPS", sps, state.global_step)

        evaluate_and_save_checkpoint(
            agent=agent,
            state=state,
            checkpoint_dir=checkpoint_dir,
            writer=writer,
            window_size=t_config.checkpoint_window,
        )

    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
