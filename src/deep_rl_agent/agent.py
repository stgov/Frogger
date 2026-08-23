from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import NamedTuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class AgentOutput(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor


class BaseAgent(ABC):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> AgentOutput: ...

    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def update(self, rollout_buffer: dict[str, torch.Tensor]) -> dict[str, float]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> dict: ...


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
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
    def __init__(self, in_channels: int, out_channels: int) -> None:
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
    def __init__(
        self, in_channels: int = 4, depth_channels: tuple[int, ...] = (16, 32, 32)
    ) -> None:
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
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.encoder = ImpalaCNN(in_channels=in_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            hidden_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU())
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.fc(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.get_features(x)
        return self.actor(features), self.critic(features)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(num_steps)):
        next_nonterminal = 1.0 - dones[t].float()
        next_values = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        advantages[t] = last_gae_lam

    return advantages, advantages + values


def iterate_minibatch_indices(
    batch_size: int, minibatch_size: int, device: torch.device
) -> Iterator[torch.Tensor]:
    """Yields shuffled index tensors covering the batch once (one epoch)."""
    indices = torch.randperm(batch_size, device=device)
    for start in range(0, batch_size, minibatch_size):
        yield indices[start : start + minibatch_size]


def clipped_policy_loss(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float
) -> torch.Tensor:
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    return torch.max(unclipped, clipped).mean()


def clipped_value_loss(
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    unclipped = (new_value - returns) ** 2
    clipped_value = old_value + torch.clamp(new_value - old_value, -clip_coef, clip_coef)
    clipped = (clipped_value - returns) ** 2
    return 0.5 * torch.max(unclipped, clipped).mean()


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Cuanto de la varianza del retorno explica la funcion de valor (1.0 =
    perfecto, 0.0 = tan bueno como predecir la media, negativo = peor que
    eso). Bajo explained_variance es la señal clásica de que el value head
    es el cuello de botella para la eficiencia por muestra: si no predice
    bien el retorno, GAE genera ventajas ruidosas y la política aprende
    más lento con la misma cantidad de datos."""
    var_returns = returns.var()
    if var_returns == 0:
        return float("nan")
    return (1 - (returns - values).var() / var_returns).item()


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
        ent_coef: float = 0.025,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        target_kl: float | None = None,
    ) -> None:
        super().__init__(observation_space, action_space, device)

        self.in_channels = observation_space.shape[0]
        self.num_actions = action_space.n

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl

        self.network: ActorCriticNetwork = ActorCriticNetwork(
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

        with torch.no_grad():
            advantages, returns = compute_gae(
                rewards, values, dones, next_value, self.gamma, self.gae_lambda
            )

        b_obs = obs.reshape((-1, *self.observation_space.shape))
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropy_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kls: list[float] = []

        for _ in range(self.update_epochs):
            for mb_inds in iterate_minibatch_indices(batch_size, minibatch_size, self.device):
                output = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                log_ratio = output.log_probs - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss = clipped_policy_loss(ratio, mb_advantages, self.clip_coef)
                v_loss = clipped_value_loss(
                    output.values, b_values[mb_inds], b_returns[mb_inds], self.clip_coef
                )
                entropy_loss = output.entropies.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Estimador de baja varianza de KL(old || new), ver http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )
                    approx_kls.append(approx_kl.item())

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

            if self.target_kl is not None and (sum(approx_kls) / len(approx_kls)) > self.target_kl:
                # Reusar más este rollout ya no ayuda (la politica se alejo
                # demasiado de la que genero los datos) — parar temprano
                # evita gastar gradientes en datos efectivamente off-policy.
                break

        with torch.no_grad():
            explained_var = _explained_variance(b_values, b_returns)

        return {
            "policy_loss": sum(pg_losses) / len(pg_losses),
            "value_loss": sum(v_losses) / len(v_losses),
            "entropy": sum(entropy_losses) / len(entropy_losses),
            "clip_fraction": sum(clip_fractions) / len(clip_fractions),
            "approx_kl": sum(approx_kls) / len(approx_kls),
            "explained_variance": explained_var,
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
                "best_avg_return": best_avg,
            },
            path,
        )

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
