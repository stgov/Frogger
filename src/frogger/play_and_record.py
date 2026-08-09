"""
Deja jugar Frogger con el teclado y graba las transiciones (obs, accion)
usando EXACTAMENTE el mismo pipeline de preprocesamiento que el
entrenamiento (frame skip, grayscale, resize a 84x84, frame stack de 4),
para que las demostraciones sean directamente utilizables por
bootstrap_bc.py sin ningun desajuste entre lo que ve el humano y lo que
ve el agente.

La ventana muestra una imagen a COLOR y a tamaño legible (usa
render_mode="rgb_array" del emulador para mostrar), pero lo que se graba
es la observacion real (gris, 84x84, apilada x4) que recibe la politica.

Controles: flechas de direccion. ESC o cerrar la ventana corta la sesion
(se guarda lo grabado hasta ese punto).

Uso:
    pip install pygame
    python play_and_record.py --episodes 5 --out demonstrations
"""

import argparse
import os

import ale_py
import gymnasium as gym
import numpy as np
import pygame

from frogger.config import AtariPreprocessingArgs, EnvSetup, GeneralConfig, TrainingConfig

t_config = TrainingConfig()

UPSCALE = 4  # factor de agrandado solo para visualizacion, no afecta la observacion grabada


def build_play_env() -> gym.Env:
    gym.register_envs(ale_py)
    env = gym.make(**GeneralConfig(render_mode="rgb_array"), **EnvSetup())
    env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingArgs(frame_skip=1, noop_max=0))
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def build_keymap(env: gym.Env) -> dict[int, int]:
    """Mapea teclas de pygame a indices de accion usando los nombres reales
    del action set del juego (no se hardcodean indices que podrian no
    coincidir segun la config)."""
    meanings = env.unwrapped.get_action_meanings()
    name_to_key = {
        "UP": pygame.K_UP,
        "DOWN": pygame.K_DOWN,
        "LEFT": pygame.K_LEFT,
        "RIGHT": pygame.K_RIGHT,
        "FIRE": pygame.K_SPACE,
    }
    keymap: dict[int, int] = {}
    for idx, name in enumerate(meanings):
        key = name_to_key.get(name)
        if key is not None:
            keymap[key] = idx
    return keymap


def action_from_keys(pressed, keymap: dict[int, int], noop_action: int) -> int:
    for key, action in keymap.items():
        if pressed[key]:
            return action
    return noop_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=5, help="Cantidad de demostraciones a grabar"
    )
    parser.add_argument("--out", type=str, default="demonstrations", help="Carpeta de salida")
    parser.add_argument(
        "--fps", type=int, default=30, help="Velocidad de juego (frames por segundo)"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    env = build_play_env()

    noop_action = 0
    try:
        noop_action = env.unwrapped.get_action_meanings().index("NOOP")
    except ValueError:
        pass

    pygame.init()
    obs, info = env.reset(seed=t_config.seed)
    frame = env.render()
    h, w = frame.shape[0] * UPSCALE, frame.shape[1] * UPSCALE
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Frogger")
    clock = pygame.time.Clock()

    keymap = build_keymap(env)

    episode_idx = 0
    quit_requested = False

    while episode_idx < args.episodes and not quit_requested:
        obs, info = env.reset()
        ep_obs: list[np.ndarray] = []
        ep_actions: list[int] = []
        done = False

        print(f"\n=== Demostracion {episode_idx + 1}/{args.episodes} — a jugar ===")

        while not done and not quit_requested:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True

            pressed = pygame.key.get_pressed()
            action = action_from_keys(pressed, keymap, noop_action)

            # Se graba la observacion ANTES del step: es la que el humano
            # usó para decidir esa accion (asi queda como par (obs, accion)
            # apto para behavior cloning).
            ep_obs.append(np.array(obs, dtype=np.uint8, copy=True))
            ep_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surface = pygame.transform.scale(surface, (w, h))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)

        if quit_requested and len(ep_obs) == 0:
            break

        out_path = os.path.join(args.out, f"demo_{episode_idx:02d}.npz")
        np.savez_compressed(
            out_path,
            obs=np.stack(ep_obs, axis=0),
            actions=np.array(ep_actions, dtype=np.int64),
        )
        print(f"Guardado {out_path} ({len(ep_actions)} pasos)")
        episode_idx += 1

    env.close()
    pygame.quit()
    print(f"\nListo. {episode_idx} demostraciones guardadas en '{args.out}/'.")


if __name__ == "__main__":
    main()
