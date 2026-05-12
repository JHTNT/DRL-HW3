"""Smoke test for the cleaned GridWorld baseline.

This script verifies that each mode can be initialized, flattened into a DQN
state vector, and played by a random policy without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gridworld import ACTIONS, Gridworld
from src.train_utils import evaluate_random_policy, save_json, set_seed


def inspect_mode(mode: str) -> dict[str, object]:
    env = Gridworld(size=4, mode=mode)
    state = env.state()

    print(f"\n[{mode}] initial board:")
    print(env.display())
    print(f"state shape: {state.shape}")
    print(f"initial reward: {env.reward()}")

    result = evaluate_random_policy(mode=mode, episodes=20, max_steps=50, size=4)
    print(
        "random policy: "
        f"win_rate={result.win_rate:.2f}, "
        f"loss_rate={result.loss_rate:.2f}, "
        f"avg_reward={result.average_reward:.2f}, "
        f"avg_steps={result.average_steps:.2f}"
    )

    return {
        "mode": mode,
        "state_shape": list(state.shape),
        "actions": list(ACTIONS),
        "random_policy": result,
    }


def main() -> None:
    set_seed(7)
    summaries = [inspect_mode(mode) for mode in ["static", "player", "random"]]
    save_json({"modes": summaries}, "results/logs/gridworld_baseline_check.json")
    print("\nSaved results/logs/gridworld_baseline_check.json")


if __name__ == "__main__":
    main()
