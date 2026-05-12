"""HW3-1: train DQN with experience replay on static GridWorld."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from src.gridworld import ACTIONS, Gridworld
from src.models import DQN
from src.replay_buffer import ReplayBuffer
from src.train_utils import (
    ensure_dirs,
    epsilon_by_episode,
    evaluate_torch_q_network,
    get_torch_device,
    save_curve_plot,
    save_json,
    set_seed,
)


def optimize_model(
    model: DQN,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    """Run one replay-buffer optimization step."""

    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = model(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q_values = model(next_states_t).max(dim=1).values
        targets = rewards_t + gamma * next_q_values * (1.0 - dones_t)

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    ensure_dirs()
    device = get_torch_device(force_cpu=args.cpu)

    model = DQN(state_dim=64, action_dim=len(ACTIONS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=args.replay_size)

    losses: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    start_time = time.perf_counter()

    for episode in range(args.episodes):
        env = Gridworld(size=4, mode="static")
        total_reward = 0.0
        epsilon = epsilon_by_episode(
            episode,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
        )

        for step in range(1, args.max_steps + 1):
            state = env.state()

            if random.random() < epsilon:
                action = int(np.random.randint(0, len(ACTIONS)))
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = int(torch.argmax(model(state_t), dim=1).item())

            env.make_move(action)
            reward = float(env.reward())
            next_state = env.state()
            done = reward in {-10.0, 10.0}
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            loss = optimize_model(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device=device,
            )
            if loss is not None:
                losses.append(loss)

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        if (episode + 1) % args.log_interval == 0:
            recent_rewards = episode_rewards[-args.log_interval :]
            recent_loss = float(np.mean(losses[-args.log_interval :])) if losses else float("nan")
            print(
                f"episode={episode + 1:04d} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={np.mean(recent_rewards):6.2f} "
                f"avg_loss={recent_loss:8.4f} "
                f"buffer={len(replay_buffer):04d}"
            )

    training_time = time.perf_counter() - start_time
    evaluation = evaluate_torch_q_network(
        model,
        mode="static",
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        device=str(device),
    )

    model_path = "results/models/dqn_replay_static.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "evaluation": evaluation,
        },
        model_path,
    )

    loss_plot = "results/figures/dqn_replay_static_loss.png"
    reward_plot = "results/figures/dqn_replay_static_reward.png"
    save_curve_plot(losses, loss_plot, "Replay DQN Static Loss", "MSE loss", rolling_window=50)
    save_curve_plot(
        episode_rewards,
        reward_plot,
        "Replay DQN Static Episode Reward",
        "episode reward",
        rolling_window=min(50, max(1, len(episode_rewards) // 5)),
    )

    summary = {
        "algorithm": "dqn_replay",
        "mode": "static",
        "episodes": args.episodes,
        "training_time_seconds": training_time,
        "final_epsilon": epsilon_by_episode(
            args.episodes - 1,
            args.epsilon_start,
            args.epsilon_final,
            args.epsilon_decay,
        ),
        "average_training_reward_last_50": float(np.mean(episode_rewards[-50:])),
        "average_training_steps_last_50": float(np.mean(episode_steps[-50:])),
        "evaluation": evaluation,
        "model_path": model_path,
        "loss_plot": loss_plot,
        "reward_plot": reward_plot,
    }
    save_json(summary, "results/logs/dqn_replay_static_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=164)
    parser.add_argument("--replay-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=150.0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    print("\nTraining complete.")
    print(f"Evaluation win rate: {summary['evaluation'].win_rate:.2f}")
    print("Saved results/logs/dqn_replay_static_summary.json")


if __name__ == "__main__":
    main()
