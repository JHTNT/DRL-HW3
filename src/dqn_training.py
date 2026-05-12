"""Reusable DQN training loop for HW3 variants."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .gridworld import ACTIONS, Gridworld
from .models import DQN, DuelingDQN
from .replay_buffer import ReplayBuffer
from .train_utils import (
    EvaluationResult,
    ensure_dirs,
    epsilon_by_episode,
    evaluate_torch_q_network,
    get_torch_device,
    save_curve_plot,
    save_json,
    set_seed,
)

AlgorithmName = Literal["dqn", "double_dqn", "dueling_dqn"]


@dataclass
class DQNTrainingConfig:
    """Configuration for a DQN-family training run."""

    algorithm: AlgorithmName
    mode: str
    output_prefix: str
    episodes: int = 1000
    max_steps: int = 50
    gamma: float = 0.9
    lr: float = 1e-3
    hidden_dim: int = 164
    replay_size: int = 5000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: float = 250.0
    target_sync_freq: int = 100
    eval_episodes: int = 100
    log_interval: int = 100
    seed: int = 7
    force_cpu: bool = False
    use_huber_loss: bool = True
    gradient_clip: float | None = 5.0


def build_model(algorithm: AlgorithmName, hidden_dim: int) -> nn.Module:
    """Create the model architecture for a DQN-family algorithm."""

    if algorithm == "dueling_dqn":
        return DuelingDQN(state_dim=64, action_dim=len(ACTIONS), hidden_dim=hidden_dim)
    return DQN(state_dim=64, action_dim=len(ACTIONS), hidden_dim=hidden_dim)


def _compute_targets(
    algorithm: AlgorithmName,
    online_model: nn.Module,
    target_model: nn.Module,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute DQN or Double DQN targets."""

    with torch.no_grad():
        if algorithm == "double_dqn":
            next_actions = online_model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_model(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = target_model(next_states).max(dim=1).values
        return rewards + gamma * next_q_values * (1.0 - dones)


def optimize_model(
    algorithm: AlgorithmName,
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    gradient_clip: float | None,
) -> float | None:
    """Run one optimization step from replay memory."""

    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = online_model(states_t).gather(1, actions_t).squeeze(1)
    targets = _compute_targets(
        algorithm=algorithm,
        online_model=online_model,
        target_model=target_model,
        rewards=rewards_t,
        next_states=next_states_t,
        dones=dones_t,
        gamma=gamma,
    )

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), gradient_clip)
    optimizer.step()
    return float(loss.item())


def train_dqn_variant(config: DQNTrainingConfig) -> dict[str, Any]:
    """Train a DQN-family agent and save model, logs, and figures."""

    set_seed(config.seed)
    ensure_dirs()
    device = get_torch_device(force_cpu=config.force_cpu)

    online_model = build_model(config.algorithm, config.hidden_dim).to(device)
    target_model = build_model(config.algorithm, config.hidden_dim).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=config.lr)
    loss_fn: nn.Module = nn.SmoothL1Loss() if config.use_huber_loss else nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=config.replay_size)

    losses: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    win_rates: list[float] = []
    global_step = 0
    start_time = time.perf_counter()

    for episode in range(config.episodes):
        env = Gridworld(size=4, mode=config.mode)
        total_reward = 0.0
        epsilon = epsilon_by_episode(
            episode,
            epsilon_start=config.epsilon_start,
            epsilon_final=config.epsilon_final,
            epsilon_decay=config.epsilon_decay,
        )
        won = False

        for step in range(1, config.max_steps + 1):
            global_step += 1
            state = env.state()

            if random.random() < epsilon:
                action = int(np.random.randint(0, len(ACTIONS)))
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = int(torch.argmax(online_model(state_t), dim=1).item())

            env.make_move(action)
            reward = float(env.reward())
            next_state = env.state()
            done = reward in {-10.0, 10.0}
            total_reward += reward
            won = reward == 10.0

            replay_buffer.push(state, action, reward, next_state, done)
            loss = optimize_model(
                algorithm=config.algorithm,
                online_model=online_model,
                target_model=target_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                replay_buffer=replay_buffer,
                batch_size=config.batch_size,
                gamma=config.gamma,
                device=device,
                gradient_clip=config.gradient_clip,
            )
            if loss is not None:
                losses.append(loss)

            if global_step % config.target_sync_freq == 0:
                target_model.load_state_dict(online_model.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step)
        win_rates.append(1.0 if won else 0.0)

        if (episode + 1) % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval :]
            recent_wins = win_rates[-config.log_interval :]
            recent_loss = float(np.mean(losses[-config.log_interval :])) if losses else float("nan")
            print(
                f"{config.algorithm} episode={episode + 1:04d} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={np.mean(recent_rewards):6.2f} "
                f"win_rate={np.mean(recent_wins):.2f} "
                f"avg_loss={recent_loss:8.4f} "
                f"buffer={len(replay_buffer):04d}"
            )

    target_model.load_state_dict(online_model.state_dict())
    training_time = time.perf_counter() - start_time
    evaluation: EvaluationResult = evaluate_torch_q_network(
        online_model,
        mode=config.mode,
        episodes=config.eval_episodes,
        max_steps=config.max_steps,
        device=str(device),
    )

    model_path = f"results/models/{config.output_prefix}.pt"
    torch.save(
        {
            "model_state_dict": online_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "config": config.__dict__,
            "evaluation": evaluation,
        },
        model_path,
    )

    loss_plot = f"results/figures/{config.output_prefix}_loss.png"
    reward_plot = f"results/figures/{config.output_prefix}_reward.png"
    win_rate_plot = f"results/figures/{config.output_prefix}_win_rate.png"
    save_curve_plot(losses, loss_plot, f"{config.output_prefix} Loss", "loss", rolling_window=50)
    save_curve_plot(
        episode_rewards,
        reward_plot,
        f"{config.output_prefix} Episode Reward",
        "episode reward",
        rolling_window=min(50, max(1, len(episode_rewards) // 5)),
    )
    save_curve_plot(
        win_rates,
        win_rate_plot,
        f"{config.output_prefix} Training Win Indicator",
        "win indicator",
        rolling_window=min(100, max(1, len(win_rates) // 5)),
    )

    summary = {
        "algorithm": config.algorithm,
        "mode": config.mode,
        "episodes": config.episodes,
        "training_time_seconds": training_time,
        "global_steps": global_step,
        "final_epsilon": epsilon_by_episode(
            config.episodes - 1,
            config.epsilon_start,
            config.epsilon_final,
            config.epsilon_decay,
        ),
        "average_training_reward_last_100": float(np.mean(episode_rewards[-100:])),
        "average_training_steps_last_100": float(np.mean(episode_steps[-100:])),
        "training_win_rate_last_100": float(np.mean(win_rates[-100:])),
        "evaluation": evaluation,
        "model_path": model_path,
        "loss_plot": loss_plot,
        "reward_plot": reward_plot,
        "win_rate_plot": win_rate_plot,
    }

    summary_path = Path(f"results/logs/{config.output_prefix}_summary.json")
    save_json(summary, summary_path)
    return summary
