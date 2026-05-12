"""Create a comparison table/plot for HW3-2 player-mode experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.train_utils import save_json

SUMMARY_FILES = {
    "Double DQN": Path("results/logs/double_dqn_player_summary.json"),
    "Dueling DQN": Path("results/logs/dueling_dqn_player_summary.json"),
}


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    rows: list[dict[str, object]] = []
    for name, path in SUMMARY_FILES.items():
        summary = load_summary(path)
        if summary is None:
            print(f"Skip missing file: {path}")
            continue
        evaluation = summary["evaluation"]
        assert isinstance(evaluation, dict)
        rows.append(
            {
                "method": name,
                "episodes": summary["episodes"],
                "final_win_rate": evaluation["win_rate"],
                "average_reward": evaluation["average_reward"],
                "average_steps": evaluation["average_steps"],
                "training_win_rate_last_100": summary["training_win_rate_last_100"],
                "training_time_seconds": summary["training_time_seconds"],
            }
        )

    if not rows:
        raise SystemExit("No HW3-2 summaries found. Train Double DQN and/or Dueling DQN first.")

    save_json({"rows": rows}, "results/logs/hw3_2_player_comparison.json")

    labels = [str(row["method"]) for row in rows]
    win_rates = [float(row["final_win_rate"]) for row in rows]
    rewards = [float(row["average_reward"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, win_rates, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Evaluation Win Rate")
    axes[0].set_ylabel("win rate")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, rewards, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[1].set_title("Evaluation Average Reward")
    axes[1].set_ylabel("average reward")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("HW3-2 Player Mode Comparison")
    fig.tight_layout()
    output_path = Path("results/figures/hw3_2_player_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved results/logs/hw3_2_player_comparison.json")
    print("Saved results/figures/hw3_2_player_comparison.png")


if __name__ == "__main__":
    main()
