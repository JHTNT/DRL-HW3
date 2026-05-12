"""Rainbow-style DQN components for random-mode GridWorld.

This module implements Rainbow DQN components for the HW3 GridWorld setting:

- dueling Q-network
- Double DQN targets, computed in the training script
- NoisyNet linear layers
- prioritized experience replay
- n-step returns
- C51 distributional value head

The code intentionally avoids extra dependencies so it can run with the same
PyTorch-only environment used by the previous homework parts.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterator

import numpy as np
import torch
from torch import nn

from .replay_buffer import Transition


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet linear layer."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.weight_epsilon: torch.Tensor
        self.bias_epsilon: torch.Tensor
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features), persistent=False)
        self.register_buffer("bias_epsilon", torch.empty(out_features), persistent=False)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize learnable mean and noise-scale parameters."""

        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        sigma = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self) -> None:
        """Sample fresh factorized Gaussian noise."""

        epsilon_in = self._scale_noise(self.in_features, self.weight_mu.device)
        epsilon_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """Dueling Q-network with optional NoisyNet layers."""

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 164,
        use_noisy: bool = True,
        noisy_sigma: float = 0.5,
    ):
        super().__init__()
        self.use_noisy = use_noisy
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        linear = self._make_linear
        self.value_stream = nn.Sequential(
            linear(hidden_dim, hidden_dim, use_noisy, noisy_sigma),
            nn.ReLU(),
            linear(hidden_dim, 1, use_noisy, noisy_sigma),
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_dim, hidden_dim, use_noisy, noisy_sigma),
            nn.ReLU(),
            linear(hidden_dim, action_dim, use_noisy, noisy_sigma),
        )

    @staticmethod
    def _make_linear(
        in_features: int,
        out_features: int,
        use_noisy: bool,
        sigma_init: float,
    ) -> nn.Module:
        if use_noisy:
            return NoisyLinear(in_features, out_features, sigma_init=sigma_init)
        return nn.Linear(in_features, out_features)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers."""

        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class C51RainbowDQN(nn.Module):
    """Dueling distributional Rainbow network with optional NoisyNet layers.

    ``forward`` returns expected Q-values so the model can be evaluated by the
    same greedy policy helper used by the non-distributional DQN variants.
    ``dist`` returns the categorical action-value distributions used by C51.
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 164,
        atom_size: int = 51,
        v_min: float = -50.0,
        v_max: float = 10.0,
        use_noisy: bool = True,
        noisy_sigma: float = 0.5,
    ):
        super().__init__()
        if atom_size <= 1:
            raise ValueError("atom_size must be greater than 1 for C51.")
        if v_min >= v_max:
            raise ValueError("v_min must be smaller than v_max.")

        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.use_noisy = use_noisy
        self.register_buffer("support", torch.linspace(v_min, v_max, atom_size))

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        linear = RainbowDQN._make_linear
        self.value_stream = nn.Sequential(
            linear(hidden_dim, hidden_dim, use_noisy, noisy_sigma),
            nn.ReLU(),
            linear(hidden_dim, atom_size, use_noisy, noisy_sigma),
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_dim, hidden_dim, use_noisy, noisy_sigma),
            nn.ReLU(),
            linear(hidden_dim, action_dim * atom_size, use_noisy, noisy_sigma),
        )

    @property
    def delta_z(self) -> float:
        """Distance between adjacent C51 atoms."""

        return (self.v_max - self.v_min) / (self.atom_size - 1)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers."""

        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits with shape ``(batch, actions, atoms)``."""

        features = self.feature(x)
        values = self.value_stream(features).view(-1, 1, self.atom_size)
        advantages = self.advantage_stream(features).view(-1, self.action_dim, self.atom_size)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return categorical action-value distributions."""

        probabilities = torch.softmax(self.logits(x), dim=2)
        return probabilities.clamp(min=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.dist(x)
        return torch.sum(probabilities * self.support.view(1, 1, -1), dim=2)


PrioritizedSample = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


class PrioritizedReplayBuffer:
    """Proportional prioritized replay buffer."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self._buffer: list[Transition] = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._position = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def max_priority(self) -> float:
        """Return the current maximum priority for newly inserted samples."""

        if not self._buffer:
            return 1.0
        max_priority = float(self._priorities[: len(self._buffer)].max())
        return max(max_priority, 1.0)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float | None = None,
    ) -> None:
        """Append one transition with maximum priority by default."""

        new_priority = float(priority if priority is not None else self.max_priority)
        transition = Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            done=bool(done),
        )
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._priorities[self._position] = max(new_priority, self.priority_epsilon)
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> PrioritizedSample:
        """Sample a prioritized mini-batch with importance-sampling weights."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more items than currently stored.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")

        priorities = self._priorities[: len(self._buffer)]
        scaled_priorities = np.power(priorities, self.alpha)
        priority_sum = scaled_priorities.sum()
        if not np.isfinite(priority_sum) or priority_sum <= 0:
            probabilities = np.full(len(self._buffer), 1.0 / len(self._buffer), dtype=np.float32)
        else:
            probabilities = scaled_priorities / priority_sum

        indices = np.random.choice(len(self._buffer), size=batch_size, replace=True, p=probabilities)
        transitions = [self._buffer[index] for index in indices]

        weights = np.power(len(self._buffer) * probabilities[indices], -beta)
        weights /= weights.max()

        states = np.concatenate([transition.state.reshape(1, -1) for transition in transitions], axis=0)
        actions = np.asarray([transition.action for transition in transitions], dtype=np.int64)
        rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
        next_states = np.concatenate(
            [transition.next_state.reshape(1, -1) for transition in transitions], axis=0
        )
        dones = np.asarray([transition.done for transition in transitions], dtype=np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights.astype(np.float32),
            indices.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities after a learning step."""

        for index, priority in zip(indices, priorities, strict=True):
            self._priorities[int(index)] = float(max(priority, self.priority_epsilon))


class NStepTransitionBuffer:
    """Accumulate one-step transitions into n-step return transitions."""

    def __init__(self, n_step: int, gamma: float):
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        self.n_step = n_step
        self.gamma = gamma
        self._buffer: Deque[Transition] = deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Transition | None:
        """Push one step and return an n-step transition when available."""

        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
            )
        )
        if len(self._buffer) < self.n_step and not done:
            return None
        return self._pop_transition()

    def flush(self) -> Iterator[Transition]:
        """Yield remaining shorter-horizon transitions at episode end."""

        while self._buffer:
            yield self._pop_transition()

    def _pop_transition(self) -> Transition:
        transition = self._build_transition()
        self._buffer.popleft()
        return transition

    def _build_transition(self) -> Transition:
        reward = 0.0
        next_state = self._buffer[0].next_state
        done = self._buffer[0].done

        for step, transition in enumerate(list(self._buffer)[: self.n_step]):
            reward += (self.gamma**step) * transition.reward
            next_state = transition.next_state
            done = transition.done
            if transition.done:
                break

        first = self._buffer[0]
        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=next_state,
            done=done,
        )