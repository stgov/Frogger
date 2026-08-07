import os
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from frogger.agent import ImpalaPPOAgent
from frogger.config import (
    AtariPreprocessingArgs,
    EnvSetup,
    GeneralConfig,
    TrainingConfig,
)

t_config = TrainingConfig()


def make_env(seed_offset: int = 0):
    def thunk() -> gym.core.Env:
        gym.register_envs(ale_py)
        env = gym.make(**GeneralConfig(), **EnvSetup())
        env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingArgs())
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.action_space.seed(t_config.seed + seed_offset)
        return env

    return thunk


def main() -> None:
    random.seed(t_config.seed)
    np.random.seed(t_config.seed)
    torch.manual_seed(t_config.seed)
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    # Carpetas independientes por run
    checkpoint_dir = os.path.join("checkpoints", t_config.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(rf"runs/{t_config.run_name}")

    recent_returns: deque[float] = deque(maxlen=t_config.checkpoint_window)
    best_avg_return: float = -float("inf")
    global_step = 0
    start_update = 1

    # Inicializar Ambientes Vectorizados
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

    # -------------------------------------------------------------------------
    # Cargar Run Anterior (si se especifica en la config)
    # -------------------------------------------------------------------------

    # TODO: define helper function, focus on readability
    if t_config.resume_checkpoint and os.path.exists(t_config.resume_checkpoint):
        print(f"Cargando checkpoint desde: {t_config.resume_checkpoint}")
        checkpoint_data = agent.load(t_config.resume_checkpoint)
        global_step = checkpoint_data.get("global_step", 0)
        saved_returns = checkpoint_data.get("recent_returns", [])
        recent_returns.extend(saved_returns)
        best_avg_return = checkpoint_data.get("best_avg_return", -float("inf"))

        # Recalcular desde qué update reanudar
        start_update = (global_step // (t_config.num_envs * t_config.num_steps)) + 1
        print(f"Reanudando en el paso global {global_step} (Update {start_update})")

    # Buffers
    obs_shape = envs.single_observation_space.shape
    buffer_dim = (t_config.num_steps, t_config.num_envs)

    obs_buffer = torch.zeros((*buffer_dim, *obs_shape), dtype=torch.uint8, device=device)
    actions_buffer = torch.zeros(buffer_dim, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(buffer_dim, device=device)
    rewards_buffer = torch.zeros(buffer_dim, device=device)
    dones_buffer = torch.zeros(buffer_dim, device=device)
    values_buffer = torch.zeros(buffer_dim, device=device)

    initial_global_step = global_step
    start_time = time.time()
    obs, info = envs.reset(seed=t_config.seed)

    # Loop de Entrenamiento
    for update in range(start_update, t_config.total_updates + 1):
        for step in range(t_config.num_steps):
            global_step += t_config.num_envs
            obs_tensor = torch.tensor(obs, device=device)
            obs_buffer[step] = obs_tensor

            with torch.no_grad():
                output = agent.get_action_and_value(obs_tensor)

            actions_buffer[step] = output.actions
            logprobs_buffer[step] = output.log_probs
            values_buffer[step] = output.values.flatten()

            next_obs, rewards, terminations, truncations, infos = envs.step(
                output.actions.cpu().numpy()
            )
            dones = np.logical_or(terminations, truncations)

            rewards_buffer[step] = torch.tensor(rewards, device=device)
            dones_buffer[step] = torch.tensor(dones, device=device)

            if "episode" in infos:
                for idx, finished in enumerate(infos["_episode"]):
                    if finished:
                        ep_return = float(infos["episode"]["r"][idx])
                        ep_length = int(infos["episode"]["l"][idx])
                        recent_returns.append(ep_return)
                        writer.add_scalar("Eval/episodic_return", ep_return, global_step)
                        writer.add_scalar("Eval/episodic_length", ep_length, global_step)

            obs = next_obs

        with torch.no_grad():
            next_obs_tensor = torch.tensor(next_obs, device=device)
            next_value = agent.get_value(next_obs_tensor)

        rollout_data = {
            "obs": obs_buffer,
            "actions": actions_buffer,
            "log_probs": logprobs_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "values": values_buffer,
            "next_value": next_value,
        }

        # TODO: define helper function, focus on readability
        metrics = agent.update(rollout_data)

        # is false? any time?
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"losses/{k}", v, global_step)

        sps = int((global_step - initial_global_step) / (time.time() - start_time))
        writer.add_scalar("Performance/SPS", sps, global_step)

        # Checkpointing por promedio móvil
        # TODO: define helper function, focus on readability
        if len(recent_returns) >= t_config.checkpoint_window:
            current_avg_return = float(np.mean(recent_returns))
            writer.add_scalar("Eval/avg_return_window", current_avg_return, global_step)

            if current_avg_return > best_avg_return:
                best_avg_return = current_avg_return
                ckpt_path = os.path.join(
                    checkpoint_dir, f"best_agent_score_{best_avg_return:.2f}.pt"
                )
                agent.save(ckpt_path, global_step, list(recent_returns))
                print(
                    f"[Paso {global_step}] Nuevo récord ({best_avg_return:.2f}). Checkpoint guardado en {ckpt_path}"
                )

    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
