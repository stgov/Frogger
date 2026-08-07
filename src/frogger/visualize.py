import argparse
from dataclasses import asdict, dataclass
from typing import Literal

import ale_py
import gymnasium as gym
import torch

from frogger.agent import ImpalaPPOAgent


@dataclass
class GeneralConfig:
    obs_type: Literal["rgb", "ram", "grayscale"] = "grayscale"
    full_action_space: bool = False
    frameskip: int = 1


@dataclass
class EnvSetup:
    id: str = "ALE/Frogger-v5"
    mode: Literal[0, 1, 2] = 0
    difficulty: Literal[0, 1] = 1


def make_eval_env(
    render_mode: Literal["human", "rgb_array", "ansi", "rgb_array_list", "ansi_list"] = "human",
    config: GeneralConfig = GeneralConfig(),
    setup: EnvSetup = EnvSetup(),
):
    gym.register_envs(ale_py)

    env = gym.make(
        **asdict(config),
        **asdict(setup),
        render_mode=render_mode,
    )

    # Apply standard preprocessing wrappers (must match training pipeline)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
    )

    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def evaluate(
    checkpoint_path: str,
    episodes: int = 5,
    record_video: bool = False,
    video_folder: str = "recordings",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if record_video else "human"

    # 1. Create Environment
    env = make_eval_env(render_mode=render_mode)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="impala_frogger",
            episode_trigger=lambda ep: True,  # Record all evaluation episodes
        )

    # 2. Instantiate Agent & Load Checkpoint
    agent = ImpalaPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"Loading model checkpoint from: {checkpoint_path}")
    agent.load(checkpoint_path)
    agent.network.eval()

    # 3. Evaluation Loop
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_over = False
        total_reward = 0.0
        steps = 0

        while not episode_over:
            # Prepare observation tensor: shape (1, 4, 84, 84)
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)

            with torch.no_grad():
                # Get deterministic action logits
                logits, _ = agent.network(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            episode_over = terminated or truncated

        print(
            f"Episode {episode}/{episodes} Finished | Total Reward: {total_reward} | Steps: {steps}"
        )

    env.close()
    print("Visualization finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained RL Agent playing Frogger")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to play",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record MP4 videos instead of displaying live window",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="recordings",
        help="Output folder for recorded videos",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        record_video=args.record,
        video_folder=args.video_dir,
    )
