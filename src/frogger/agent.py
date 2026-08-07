from abc import ABC, abstractmethod
from typing import NamedTuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# -----------------------------------------------------------------------------
# Interface Protocol & Data Types
# -----------------------------------------------------------------------------
class AgentOutput(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor


class BaseAgent(ABC):
    """Abstract Base Class defining the agent contract for vector environments."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> AgentOutput:
        pass

    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, rollout_buffer: dict[str, torch.Tensor]) -> dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


# -----------------------------------------------------------------------------
# IMPALA Neural Network Backbone
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, in_channels: int = 4, depth_channels=(16, 32, 32)):
        super().__init__()
        layers = []
        cur_channels = in_channels
        for out_channels in depth_channels:
            layers.append(ConvSequence(cur_channels, out_channels))
            cur_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = torch.relu(x)
        return torch.flatten(x, start_dim=1)


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.encoder = ImpalaCNN(in_channels=in_channels)

        # Compute output linear size dynamically with dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            hidden_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU())
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Scale uint8 [0, 255] to float [0.0, 1.0]
        x = x.float() / 255.0
        return self.fc(self.encoder(x))

    def forward(self, x: torch.Tensor):
        features = self.get_features(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class ImpalaPPOAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
    ):
        super().__init__(observation_space, action_space, device)

        self.in_channels = observation_space.shape[0]
        self.num_actions = action_space.n

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches

        # Network & Optimizer
        self.network = ActorCriticNetwork(
            in_channels=self.in_channels, num_actions=self.num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> AgentOutput:
        logits, values = self.network(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return AgentOutput(
            actions=action,
            log_probs=dist.log_prob(action),
            entropies=dist.entropy(),
            values=values.squeeze(-1),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, values = self.network(obs)
        return values.squeeze(-1)

    def update(self, rollout_buffer: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = rollout_buffer["obs"]  # (NUM_STEPS, NUM_ENVS, C, H, W)
        actions = rollout_buffer["actions"]  # (NUM_STEPS, NUM_ENVS)
        logprobs = rollout_buffer["log_probs"]  # (NUM_STEPS, NUM_ENVS)
        rewards = rollout_buffer["rewards"]  # (NUM_STEPS, NUM_ENVS)
        dones = rollout_buffer["dones"]  # (NUM_STEPS, NUM_ENVS)
        values = rollout_buffer["values"]  # (NUM_STEPS, NUM_ENVS)
        next_value = rollout_buffer["next_value"]  # (NUM_ENVS,)

        num_steps, num_envs = actions.shape
        batch_size = num_steps * num_envs
        minibatch_size = batch_size // self.num_minibatches

        # 1. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones[t].float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t].float()
                    nextvalues = values[t + 1]

                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # Flatten buffers across time and environment dimensions
        b_obs = obs.reshape((-1,) + self.observation_space.shape)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 2. PPO Optimization Epochs
        b_inds = torch.arange(batch_size, device=self.device)
        clip_fractions = []

        pg_losses, v_losses, entropy_losses = [], [], []

        for epoch in range(self.update_epochs):
            # Shuffle minibatch indices
            shuffled_inds = b_inds[torch.randperm(batch_size, device=self.device)]

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = shuffled_inds[start:end]

                output = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                newlogprob = output.log_probs
                entropy = output.entropies
                newvalue = output.values

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl for monitoring
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy Loss (PPO-Clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Total Loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": sum(pg_losses) / len(pg_losses),
            "value_loss": sum(v_losses) / len(v_losses),
            "entropy": sum(entropy_losses) / len(entropy_losses),
            "clip_fraction": sum(clip_fractions) / len(clip_fractions),
        }

    def save(
        self,
        path: str,
        global_step: int = 0,
        recent_returns: list[float] | None = None,
        best_avg: float = -float("inf"),
    ) -> None:
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": global_step,
                "recent_returns": recent_returns or [],
                "best_avg_return": best_avg,  # Add this
            },
            path,
        )

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
