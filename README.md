# DRL-HW3

Homework 3 implementation workspace for DQN and DQN variants on the DRL in Action Chapter 3 GridWorld environment.

## Current status

Phase 1 baseline setup is complete and HW3-1 scripts are available:

- `reference/` keeps the original DRL in Action reference files unchanged.
- `src/gridboard.py` and `src/gridworld.py` provide importable cleaned modules.
- `src/models.py` contains PyTorch DQN models.
- `src/replay_buffer.py` contains the experience replay buffer.
- `scripts/check_gridworld_baseline.py` verifies `static`, `player`, and `random` modes.
- `scripts/train_naive_dqn_static.py` trains naive DQN for HW3-1.
- `scripts/train_dqn_replay_static.py` trains replay-buffer DQN for HW3-1.
- `scripts/train_double_dqn_player.py` trains Double DQN for HW3-2.
- `scripts/train_dueling_dqn_player.py` trains Dueling DQN for HW3-2.
- `scripts/compare_hw3_2_player.py` creates the HW3-2 comparison output.
- `scripts/train_lightning_dqn_random.py` trains an official PyTorch Lightning Dueling Double DQN for HW3-3 random mode.
- `scripts/train_rainbow_random.py` trains a Rainbow DQN with C51, Double DQN, Dueling DQN, prioritized replay, n-step returns, and NoisyNet for the HW3-4 bonus random-mode task.
- `results/` stores generated logs, figures, and model checkpoints.

## Run baseline smoke test

Use `uv` when available:

```bash
uv run python scripts/check_gridworld_baseline.py
```

If this workspace is not managed by `uv`, use the configured Python environment in VS Code to run the same script.

## Run HW3-1 training

Install dependencies with `uv`:

```bash
uv add numpy torch matplotlib
```

Train static-mode DQN variants:

```bash
uv run python scripts/train_naive_dqn_static.py
uv run python scripts/train_dqn_replay_static.py
```

For a quick smoke test:

```bash
uv run python scripts/train_dqn_replay_static.py --episodes 100 --eval-episodes 50
```

## Run HW3-2 training

Train player-mode Double DQN and Dueling DQN:

```bash
uv run python scripts/train_double_dqn_player.py
uv run python scripts/train_dueling_dqn_player.py
uv run python scripts/compare_hw3_2_player.py
```

Quick smoke test:

```bash
uv run python scripts/train_double_dqn_player.py --episodes 100 --eval-episodes 50
uv run python scripts/train_dueling_dqn_player.py --episodes 100 --eval-episodes 50
uv run python scripts/compare_hw3_2_player.py
```

## Run HW3-3 training

Train random-mode DQN with an official PyTorch Lightning `LightningModule` / `Trainer` split and training tips:

```bash
uv run python scripts/train_lightning_dqn_random.py
```

Quick smoke test:

```bash
uv run python scripts/train_lightning_dqn_random.py --episodes 50 --eval-episodes 20 --eval-interval 25 --replay-warmup 32 --batch-size 32
```

Outputs:

- `results/models/lightning_dqn_random.pt`
- `results/logs/lightning_dqn_random_summary.json`
- `results/figures/lightning_dqn_random_loss.png`
- `results/figures/lightning_dqn_random_reward.png`
- `results/figures/lightning_dqn_random_win_rate.png`

## Run HW3-4 bonus training

Train random-mode Rainbow DQN with C51 distributional learning, Double DQN, Dueling DQN, prioritized replay, n-step returns, and NoisyNet:

```bash
uv run python scripts/train_rainbow_random.py
```

Quick smoke test:

```bash
uv run python scripts/train_rainbow_random.py --episodes 50 --eval-episodes 20 --eval-interval 25 --replay-warmup 32 --batch-size 32
```

Optional ablations:

```bash
uv run python scripts/train_rainbow_random.py --no-distributional
uv run python scripts/train_rainbow_random.py --no-noisy
```

Outputs:

- `results/models/rainbow_random.pt`
- `results/logs/rainbow_random_summary.json`
- `results/figures/rainbow_random_loss.png`
- `results/figures/rainbow_random_reward.png`
- `results/figures/rainbow_random_win_rate.png`

## GridWorld summary

- State: one-hot board layers with shape `(4, 4, 4)`, flattened to `(1, 64)` for DQN.
- Actions: `0=u`, `1=d`, `2=l`, `3=r`.
- Rewards: goal `+10`, pit `-10`, otherwise `-1`.
- Modes:
  - `static`: all pieces fixed.
  - `player`: only player starts randomly.
  - `random`: all pieces start randomly.
