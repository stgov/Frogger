import argparse
from typing import Literal

import ale_py
import gymnasium as gym
import torch

from deep_rl_agent.agent import ImpalaPPOAgent
from deep_rl_agent.config import (
    AUTO_FIRE,
    AtariPreprocessingArgs,
    EnvSetup,
    GeneralConfig,
    TrainingConfig,
)
from deep_rl_agent.env_wrappers import FireResetEnv


def make_eval_env(
    render_mode: Literal["human", "rgb_array", "ansi", "rgb_array_list", "ansi_list"] = "human",
):
    gym.register_envs(ale_py)

    # pyrefly: ignore [missing-argument]
    env = gym.make(
        **GeneralConfig(render_mode=render_mode),
        **EnvSetup(),
    )

    env = FireResetEnv(env) if AUTO_FIRE else env

    # Apply standard preprocessing wrappers (must match training pipeline)
    env = gym.wrappers.AtariPreprocessing(env, **AtariPreprocessingArgs())

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

    env = make_eval_env(render_mode=render_mode)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"{TrainingConfig().run_name}",
            episode_trigger=lambda ep: True,
        )

    agent = ImpalaPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"Loading model checkpoint from: {checkpoint_path}")
    agent.load(checkpoint_path)
    agent.network.eval()

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
    parser = argparse.ArgumentParser(description="Visualize trained RL Agent playing deep_rl_agent")
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
