"""HW3-1: train a naive DQN on static GridWorld without replay buffer."""

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
from src.train_utils import (
    ensure_dirs,
    epsilon_by_episode,
    evaluate_torch_q_network,
    get_torch_device,
    save_curve_plot,
    save_json,
    set_seed,
)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    ensure_dirs()
    device = get_torch_device(force_cpu=args.cpu)

    model = DQN(state_dim=64, action_dim=len(ACTIONS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    start_time = time.perf_counter()

    for episode in range(args.episodes):
        env = Gridworld(size=4, mode="static")
        total_reward = 0.0
        step = 0
        epsilon = epsilon_by_episode(
            episode,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
        )

        for step in range(1, args.max_steps + 1):
            state_np = env.state()
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device)

            if random.random() < epsilon:
                action = int(np.random.randint(0, len(ACTIONS)))
            else:
                with torch.no_grad():
                    action = int(torch.argmax(model(state), dim=1).item())

            env.make_move(action)
            reward = float(env.reward())
            next_state = torch.as_tensor(env.state(), dtype=torch.float32, device=device)
            done = reward in {-10.0, 10.0}
            total_reward += reward

            q_values = model(state)
            target = q_values.detach().clone()
            if done:
                target_value = reward
            else:
                with torch.no_grad():
                    target_value = reward + args.gamma * float(model(next_state).max(dim=1).values.item())
            target[0, action] = target_value

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        if (episode + 1) % args.log_interval == 0:
            recent_rewards = episode_rewards[-args.log_interval :]
            print(
                f"episode={episode + 1:04d} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={np.mean(recent_rewards):6.2f} "
                f"last_steps={step:02d}"
            )

    training_time = time.perf_counter() - start_time
    evaluation = evaluate_torch_q_network(
        model,
        mode="static",
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        device=str(device),
    )

    model_path = "results/models/naive_dqn_static.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "evaluation": evaluation,
        },
        model_path,
    )

    loss_plot = "results/figures/naive_dqn_static_loss.png"
    reward_plot = "results/figures/naive_dqn_static_reward.png"
    save_curve_plot(losses, loss_plot, "Naive DQN Static Loss", "MSE loss", rolling_window=50)
    save_curve_plot(
        episode_rewards,
        reward_plot,
        "Naive DQN Static Episode Reward",
        "episode reward",
        rolling_window=min(50, max(1, len(episode_rewards) // 5)),
    )

    summary = {
        "algorithm": "naive_dqn",
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
    save_json(summary, "results/logs/naive_dqn_static_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=164)
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
    evaluation = summary["evaluation"]
    print("\nTraining complete.")
    print(f"Evaluation win rate: {getattr(evaluation, 'win_rate'):.2f}")
    print("Saved results/logs/naive_dqn_static_summary.json")


if __name__ == "__main__":
    main()
