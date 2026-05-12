"""HW3-2: train Double DQN on player-mode GridWorld."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dqn_training import DQNTrainingConfig, train_dqn_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=164)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=250.0)
    parser.add_argument("--target-sync-freq", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--mse-loss", action="store_true", help="Use MSE instead of Huber loss.")
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNTrainingConfig(
        algorithm="double_dqn",
        mode="player",
        output_prefix="double_dqn_player",
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        target_sync_freq=args.target_sync_freq,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        seed=args.seed,
        force_cpu=args.cpu,
        use_huber_loss=not args.mse_loss,
        gradient_clip=args.gradient_clip,
    )
    summary = train_dqn_variant(config)
    print("\nTraining complete.")
    print(f"Evaluation win rate: {summary['evaluation'].win_rate:.2f}")
    print("Saved results/logs/double_dqn_player_summary.json")


if __name__ == "__main__":
    main()
