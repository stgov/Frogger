"""
Bootstrap por imitation learning (behavior cloning) a partir de las
demostraciones grabadas con play_and_record.py.

Entrena la red (encoder + cabeza de accion) por supervised learning para
imitar las acciones humanas, y guarda un checkpoint en el MISMO formato
que agent.save()/agent.load() usan en train.py, para poder continuar
directamente con PPO desde ese punto (resume_checkpoint).

Nota: solo se optimiza la cabeza de accion (actor). La cabeza de valor
(critic) queda con su inicializacion aleatoria — no hay forma honesta de
supervisarla con demostraciones sueltas, y de todas formas PPO la ajusta
rapido apenas arranca el entrenamiento normal.

Con pocas demostraciones (n=5) el dataset es chico: esto es un WARM START,
no un intento de que el behavior cloning "resuelva" el juego. No entrenes
por muchas epochs — el objetivo es sesgar la politica inicial hacia
comportamiento razonable (cruzar la carretera, subirse a troncos), no
memorizar las 5 partidas.

Uso:
    python bootstrap_bc.py --demos demonstrations --epochs 30
    # luego, en config.py:
    #   resume_checkpoint = r"checkpoints/bootstrap/bc_pretrained.pt"
    # y correr train.py como siempre.
"""

import argparse
import glob
import os

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from deep_rl_agent.agent import ImpalaPPOAgent
from deep_rl_agent.config import AtariPreprocessingArgs, EnvSetup, GeneralConfig, TrainingConfig

t_config = TrainingConfig()


def load_demonstrations(demos_dir: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(demos_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron demostraciones en '{demos_dir}'. Corre play_and_record.py primero."
        )

    all_obs, all_actions = [], []
    for f in files:
        data = np.load(f)
        all_obs.append(data["obs"])
        all_actions.append(data["actions"])
        print(f"  {f}: {len(data['actions'])} pasos")

    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return obs, actions


def build_spaces() -> tuple[gym.Space, gym.Space]:
    """Crea un env temporal (sin vectorizar) solo para leer observation_space
    y action_space, sin levantar los N envs paralelos de entrenamiento."""
    gym.register_envs(ale_py)
    env = gym.make(**GeneralConfig(), **EnvSetup())
    env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingArgs())
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()
    return obs_space, act_space


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, default="demonstrations")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints/bootstrap/bc_pretrained.pt")
    args = parser.parse_args()

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    print("Cargando demostraciones...")
    obs, actions = load_demonstrations(args.demos)
    n_episodes = len(glob.glob(os.path.join(args.demos, "*.npz")))
    print(f"Total: {len(actions)} transiciones humanas de {n_episodes} episodio(s)")

    if len(actions) < 200:
        print(
            "Aviso: dataset chico. Es esperable con n=5 demostraciones — esto es "
            "un warm start, no un entrenamiento supervisado completo. Evita usar "
            "muchas epochs (riesgo de sobreajuste a estas 5 partidas puntuales)."
        )

    obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)

    obs_space, act_space = build_spaces()

    agent = ImpalaPPOAgent(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        lr=args.lr,
    )

    n = obs_t.shape[0]
    print(f"\nEntrenando behavior cloning por {args.epochs} epochs (n={n} transiciones)...")

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss, total_correct = 0.0, 0

        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_obs = obs_t[idx]
            batch_actions = actions_t[idx]

            # Solo se usan los logits del actor; la cabeza de valor no
            # recibe gradiente aca (queda para que PPO la entrene despues).
            logits, _ = agent.network(batch_obs)
            loss = F.cross_entropy(logits, batch_actions)

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), agent.max_grad_norm)
            agent.optimizer.step()

            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=-1) == batch_actions).sum().item()

        avg_loss = total_loss / n
        accuracy = total_correct / n
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  accuracy_train={accuracy:.2%}"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # global_step=0 y best_avg_return=-inf: train.py lo va a tratar como un
    # arranque fresco de entrenamiento RL, solo que con la red ya
    # sesgada hacia el comportamiento humano en vez de pesos aleatorios.
    agent.save(args.out, global_step=0, recent_returns=[], best_avg=-float("inf"))
    print(f"\nCheckpoint de bootstrap guardado en: {args.out}")
    print("Para continuar con PPO, en config.py pon:")
    print(f'    resume_checkpoint = r"{args.out}"')
    print("y corre train.py normalmente.")


if __name__ == "__main__":
    main()
