"""Experience replay utilities for DQN training."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One environment transition stored in replay memory."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


TransitionBatch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to replay memory."""

        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random mini-batch as NumPy arrays."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more items than currently stored.")

        transitions = random.sample(self._buffer, batch_size)
        states = np.concatenate([transition.state.reshape(1, -1) for transition in transitions], axis=0)
        actions = np.asarray([transition.action for transition in transitions], dtype=np.int64)
        rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
        next_states = np.concatenate(
            [transition.next_state.reshape(1, -1) for transition in transitions], axis=0
        )
        dones = np.asarray([transition.done for transition in transitions], dtype=np.float32)
        return states, actions, rewards, next_states, dones
