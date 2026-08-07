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
    AtariPreprocessingAgrs,
    EnvSetup,
    GeneralConfig,
    TrainingConfig,
)


def make_env(seed: int, seed_offset: int = 0):
    def thunk() -> gym.core.Env:
        gym.register_envs(ale_py)
        env = gym.make(**GeneralConfig(), **EnvSetup())
        env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingAgrs())
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.action_space.seed(seed + seed_offset)
        return env

    return thunk


def swap_critic_and_train(
    ckpt_path_a: str,
    ckpt_path_b: str,
    steps_to_train: int = 1_000_000,
    seed: int = 1000,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    t_config = TrainingConfig()
    t_config.seed = seed

    # Nombre único para el experimento de Swap
    run_name = f"Frogger__CriticSwap__seed_{seed}"
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(rf"runs/{run_name}")

    # Inicializar Ambientes
    env_fns = [make_env(seed, seed_offset=i) for i in range(t_config.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    # 1. Instanciar Agentes
    agent_a = ImpalaPPOAgent(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        lr=t_config.learning_rate,
        update_epochs=4,
        num_minibatches=4,
    )

    agent_b = ImpalaPPOAgent(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        lr=t_config.learning_rate,
        update_epochs=4,
        num_minibatches=4,
    )

    # 2. Cargar Checkpoints
    print(f"Cargando Red A desde: {ckpt_path_a}")
    checkpoint_a = agent_a.load(ckpt_path_a)

    print(f"Cargando Red B desde: {ckpt_path_b}")
    checkpoint_b = agent_b.load(ckpt_path_b)

    # 3. Intercambiar solo la capa Critic de B hacia A
    print("Intercambiando Critic de Red B -> Red A...")
    agent_a.network.critic.load_state_dict(agent_b.network.critic.state_dict())

    # 4. Configurar Parámetros de Entrenamiento
    start_global_step = checkpoint_a.get("global_step", 0)
    target_global_step = start_global_step + steps_to_train

    total_updates = steps_to_train // (t_config.num_envs * t_config.num_steps)

    recent_returns: deque[float] = deque(maxlen=t_config.checkpoint_window)
    best_avg_return: float = -float("inf")

    # Buffers
    obs_shape = envs.single_observation_space.shape
    buffer_dim = (t_config.num_steps, t_config.num_envs)

    obs_buffer = torch.zeros((*buffer_dim, *obs_shape), dtype=torch.uint8, device=device)
    actions_buffer = torch.zeros(buffer_dim, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(buffer_dim, device=device)
    rewards_buffer = torch.zeros(buffer_dim, device=device)
    dones_buffer = torch.zeros(buffer_dim, device=device)
    values_buffer = torch.zeros(buffer_dim, device=device)

    start_time = time.time()
    global_step = start_global_step
    obs, info = envs.reset(seed=seed)

    print(
        f"Iniciando entrenamiento del modelo con Critic intercambiado desde el paso {global_step} hasta {target_global_step} ({steps_to_train} pasos totales)..."
    )

    for update in range(1, total_updates + 1):
        for step in range(t_config.num_steps):
            global_step += t_config.num_envs
            obs_tensor = torch.tensor(obs, device=device)
            obs_buffer[step] = obs_tensor

            with torch.no_grad():
                output = agent_a.get_action_and_value(obs_tensor)

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
            next_value = agent_a.get_value(next_obs_tensor)

        rollout_data = {
            "obs": obs_buffer,
            "actions": actions_buffer,
            "log_probs": logprobs_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "values": values_buffer,
            "next_value": next_value,
        }

        metrics = agent_a.update(rollout_data)

        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"losses/{k}", v, global_step)

        sps = int((global_step - start_global_step) / (time.time() - start_time))
        writer.add_scalar("Performance/SPS", sps, global_step)

        # Checkpointing
        if len(recent_returns) >= t_config.checkpoint_window:
            current_avg_return = float(np.mean(recent_returns))
            writer.add_scalar("Eval/avg_return_window", current_avg_return, global_step)

            if current_avg_return > best_avg_return:
                best_avg_return = current_avg_return
                ckpt_path = os.path.join(
                    checkpoint_dir, f"best_swap_agent_score_{best_avg_return:.2f}.pt"
                )
                agent_a.save(ckpt_path, global_step, list(recent_returns))
                print(
                    f"[Paso {global_step}] Nuevo récord ({best_avg_return:.2f}). Guardado en {ckpt_path}"
                )

    writer.close()
    envs.close()
    print("Entrenamiento post-swap finalizado con éxito.")


if __name__ == "__main__":
    # Reemplaza con las rutas a tus checkpoints
    PATH_RED_A = (
        r"checkpoints/Frogger__envs_8__steps_128__lr_0.00025__seed_1000/best_agent_score_34.65.pt"
    )
    PATH_RED_B = (
        r"checkpoints/Frogger__envs_8__steps_128__lr_0.00025__seed_2000/best_agent_score_34.30.pt"
    )

    swap_critic_and_train(
        ckpt_path_a=PATH_RED_A,
        ckpt_path_b=PATH_RED_B,
        steps_to_train=1_000_000,
        seed=1000,
    )
