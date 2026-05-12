"""Shared utilities for GridWorld DQN scripts."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .gridworld import ACTIONS, Gridworld


class QModel(Protocol):
    """Minimal protocol for models used by ``evaluate_policy``."""

    def predict_action(self, state: np.ndarray) -> int:
        """Return the greedy action index for ``state``."""


@dataclass
class EvaluationResult:
    episodes: int
    win_rate: float
    loss_rate: float
    average_reward: float
    average_steps: float


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch if available."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*")
            cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_torch_device(force_cpu: bool = False) -> Any:
    """Return a PyTorch device while suppressing noisy unavailable-CUDA warnings."""

    import torch

    if force_cpu:
        return torch.device("cpu")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
        cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")


def ensure_dirs() -> None:
    """Create standard output directories used by scripts."""

    for path in [
        "results/figures",
        "results/logs",
        "results/models",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def random_action() -> int:
    """Return a random action index."""

    return int(np.random.randint(0, len(ACTIONS)))


def run_random_episode(mode: str, max_steps: int = 50, size: int = 4) -> tuple[int, int, str]:
    """Run one random-policy episode.

    Returns ``(total_reward, steps, outcome)`` where outcome is ``win``, ``loss``
    or ``timeout``.
    """

    env = Gridworld(size=size, mode=mode)
    total_reward = 0

    for step in range(1, max_steps + 1):
        env.make_move(random_action())
        reward = env.reward()
        total_reward += reward

        if reward == 10:
            return total_reward, step, "win"
        if reward == -10:
            return total_reward, step, "loss"

    return total_reward, max_steps, "timeout"


def evaluate_random_policy(
    mode: str,
    episodes: int = 100,
    max_steps: int = 50,
    size: int = 4,
) -> EvaluationResult:
    """Evaluate a random policy in a GridWorld mode."""

    rewards: list[int] = []
    steps: list[int] = []
    wins = 0
    losses = 0

    for _ in range(episodes):
        total_reward, episode_steps, outcome = run_random_episode(mode, max_steps, size)
        rewards.append(total_reward)
        steps.append(episode_steps)
        wins += outcome == "win"
        losses += outcome == "loss"

    return EvaluationResult(
        episodes=episodes,
        win_rate=wins / episodes,
        loss_rate=losses / episodes,
        average_reward=float(np.mean(rewards)),
        average_steps=float(np.mean(steps)),
    )


def epsilon_by_episode(
    episode: int,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay: float,
) -> float:
    """Exponential epsilon decay used by DQN training scripts."""

    return float(epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay))


def evaluate_torch_q_network(
    model: Any,
    mode: str,
    episodes: int = 100,
    max_steps: int = 50,
    size: int = 4,
    device: str | None = None,
) -> EvaluationResult:
    """Evaluate a PyTorch Q-network greedily in GridWorld."""

    import torch

    model_device = torch.device(device) if device else get_torch_device()
    model.to(model_device)
    model.eval()

    rewards: list[int] = []
    steps: list[int] = []
    wins = 0
    losses = 0

    with torch.no_grad():
        for _ in range(episodes):
            env = Gridworld(size=size, mode=mode)
            total_reward = 0
            outcome = "timeout"

            for step in range(1, max_steps + 1):
                state = torch.as_tensor(env.state(), dtype=torch.float32, device=model_device)
                action = int(torch.argmax(model(state), dim=1).item())
                env.make_move(action)
                reward = env.reward()
                total_reward += reward

                if reward == 10:
                    outcome = "win"
                    break
                if reward == -10:
                    outcome = "loss"
                    break

            rewards.append(total_reward)
            steps.append(step)
            wins += outcome == "win"
            losses += outcome == "loss"

    return EvaluationResult(
        episodes=episodes,
        win_rate=wins / episodes,
        loss_rate=losses / episodes,
        average_reward=float(np.mean(rewards)),
        average_steps=float(np.mean(steps)),
    )


def save_curve_plot(
    values: list[float],
    path: str | Path,
    title: str,
    ylabel: str,
    xlabel: str = "Episode",
    rolling_window: int | None = None,
) -> None:
    """Save a simple line plot, optionally with a rolling mean overlay."""

    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(values, alpha=0.35 if rolling_window else 0.9, label=ylabel)
    if rolling_window and len(values) >= rolling_window:
        kernel = np.ones(rolling_window) / rolling_window
        rolling = np.convolve(np.asarray(values, dtype=np.float32), kernel, mode="valid")
        plt.plot(
            range(rolling_window - 1, rolling_window - 1 + len(rolling)),
            rolling,
            linewidth=2,
            label=f"{rolling_window}-episode mean",
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=150)
    plt.close()


def save_json(data: dict[str, Any] | EvaluationResult, path: str | Path) -> None:
    """Save a dictionary or dataclass result as pretty JSON."""

    def default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, EvaluationResult):
        data = asdict(data)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=default)
        file.write("\n")
