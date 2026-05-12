"""HW3-3: train a PyTorch Lightning Dueling Double DQN on random GridWorld.

This version uses the official ``lightning`` package. The RL interaction loop
is implemented inside a ``LightningModule`` and executed by ``Trainer.fit``;
manual optimization is used because DQN updates depend on replay-buffer state
rather than a supervised-learning batch alone.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from src.gridworld import ACTIONS, Gridworld
from src.models import DuelingDQN
from src.replay_buffer import ReplayBuffer
from src.train_utils import (
    EvaluationResult,
    ensure_dirs,
    epsilon_by_episode,
    evaluate_random_policy,
    evaluate_torch_q_network,
    save_curve_plot,
    save_json,
    set_seed,
)


@dataclass
class RandomDQNLightningConfig:
    """Configuration for the HW3-3 random-mode Lightning training run."""

    mode: str = "random"
    output_prefix: str = "lightning_dqn_random"
    episodes: int = 3000
    max_steps: int = 50
    gamma: float = 0.9
    lr: float = 5e-4
    hidden_dim: int = 164
    replay_size: int = 10000
    batch_size: int = 64
    replay_warmup: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: float = 1000.0
    target_sync_freq: int = 200
    eval_episodes: int = 200
    eval_interval: int = 250
    random_baseline_episodes: int = 200
    log_interval: int = 100
    seed: int = 7
    force_cpu: bool = False
    use_huber_loss: bool = True
    gradient_clip: float | None = 5.0
    use_lr_scheduler: bool = True
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.95


class EpisodeIndexDataset(IterableDataset[int]):
    """Yield episode indices so Lightning's Trainer controls the fit loop."""

    def __init__(self, episodes: int):
        self.episodes = episodes

    def __iter__(self) -> Iterator[int]:
        yield from range(self.episodes)

    def __len__(self) -> int:
        return self.episodes


class RandomDQNLightningModule(L.LightningModule):
    """Official LightningModule for random-mode Dueling Double DQN."""

    def __init__(self, config: RandomDQNLightningConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.automatic_optimization = False

        self.online_model = DuelingDQN(
            state_dim=64,
            action_dim=len(ACTIONS),
            hidden_dim=config.hidden_dim,
        )
        self.target_model = DuelingDQN(
            state_dim=64,
            action_dim=len(ACTIONS),
            hidden_dim=config.hidden_dim,
        )
        self.loss_fn: nn.Module = nn.SmoothL1Loss() if config.use_huber_loss else nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=config.replay_size)

        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self.train_win_rates: list[float] = []
        self.eval_history: list[dict[str, float]] = []
        self.env_steps = 0
        self.optimization_steps = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any | None = None

        self.sync_target_network()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values from the online network."""

        return self.online_model(states)

    @property
    def current_lr(self) -> float:
        """Return current learning rate."""

        if self._optimizer is None:
            return self.config.lr
        return float(self._optimizer.param_groups[0]["lr"])

    def configure_optimizers(self) -> Any:
        """Create Adam optimizer and optional StepLR scheduler."""

        self._optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.config.lr)
        if self.config.use_lr_scheduler:
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
            return [self._optimizer], [self._scheduler]
        self._scheduler = None
        return self._optimizer

    def train_dataloader(self) -> DataLoader[int]:
        """Return a minimal episode-index dataloader for Trainer.fit."""

        return DataLoader(EpisodeIndexDataset(self.config.episodes), batch_size=None, num_workers=0)

    def sync_target_network(self) -> None:
        """Copy online-network parameters into the target network."""

        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Select an epsilon-greedy action for one environment state."""

        if random.random() < epsilon:
            return int(np.random.randint(0, len(ACTIONS)))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(torch.argmax(self.online_model(state_t), dim=1).item())

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run one full GridWorld episode and replay-buffer optimization steps."""

        episode = int(batch.item()) if torch.is_tensor(batch) else int(batch)
        total_reward, episode_step, won = self._run_episode(episode)

        self.episode_rewards.append(total_reward)
        self.episode_steps.append(episode_step)
        self.train_win_rates.append(1.0 if won else 0.0)

        self._maybe_evaluate(episode)
        self._maybe_log_progress(episode)

        reward_tensor = torch.tensor(total_reward, dtype=torch.float32, device=self.device)
        self.log("train/episode_reward", reward_tensor, on_step=True, prog_bar=False)
        self.log("train/win", float(won), on_step=True, prog_bar=False)
        return reward_tensor

    def _run_episode(self, episode: int) -> tuple[float, int, bool]:
        env = Gridworld(size=4, mode=self.config.mode)
        total_reward = 0.0
        episode_step = 0
        won = False
        epsilon = epsilon_by_episode(
            episode,
            epsilon_start=self.config.epsilon_start,
            epsilon_final=self.config.epsilon_final,
            epsilon_decay=self.config.epsilon_decay,
        )

        for step in range(1, self.config.max_steps + 1):
            episode_step = step
            self.env_steps += 1
            state = env.state()
            action = self.select_action(state, epsilon)

            env.make_move(action)
            reward = float(env.reward())
            next_state = env.state()
            done = reward in {-10.0, 10.0}
            total_reward += reward
            won = reward == 10.0

            self.replay_buffer.push(state, action, reward, next_state, done)
            step_result = self._optimize_from_replay()
            if step_result is not None:
                self.losses.append(step_result["loss"])
                self.learning_rates.append(step_result["lr"])
                self.optimization_steps += 1

            if self.env_steps % self.config.target_sync_freq == 0:
                self.sync_target_network()

            if done:
                break

        return total_reward, episode_step, won

    def _optimize_from_replay(self) -> dict[str, float] | None:
        min_replay_size = max(self.config.batch_size, self.config.replay_warmup)
        if len(self.replay_buffer) < min_replay_size:
            return None

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_model(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_model(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.config.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.config.gradient_clip)
        optimizer.step()

        if self.config.use_lr_scheduler:
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
            if scheduler is not None:
                scheduler.step()

        return {"loss": float(loss.item()), "lr": self.current_lr}

    def _maybe_evaluate(self, episode: int) -> None:
        should_eval = self.config.eval_interval > 0 and (episode + 1) % self.config.eval_interval == 0
        if not should_eval:
            return

        eval_result = evaluate_torch_q_network(
            self.online_model,
            mode=self.config.mode,
            episodes=self.config.eval_episodes,
            max_steps=self.config.max_steps,
            device=str(self.device),
        )
        self.eval_history.append(
            {
                "episode": float(episode + 1),
                "win_rate": eval_result.win_rate,
                "loss_rate": eval_result.loss_rate,
                "average_reward": eval_result.average_reward,
                "average_steps": eval_result.average_steps,
            }
        )
        self.online_model.train()
        self.target_model.eval()

    def _maybe_log_progress(self, episode: int) -> None:
        if (episode + 1) % self.config.log_interval != 0:
            return

        recent_rewards = self.episode_rewards[-self.config.log_interval :]
        recent_wins = self.train_win_rates[-self.config.log_interval :]
        recent_loss = float(np.mean(self.losses[-self.config.log_interval :])) if self.losses else float("nan")
        epsilon = epsilon_by_episode(
            episode,
            epsilon_start=self.config.epsilon_start,
            epsilon_final=self.config.epsilon_final,
            epsilon_decay=self.config.epsilon_decay,
        )
        print(
            f"lightning random episode={episode + 1:04d} "
            f"epsilon={epsilon:.3f} "
            f"avg_reward={np.mean(recent_rewards):6.2f} "
            f"win_rate={np.mean(recent_wins):.2f} "
            f"avg_loss={recent_loss:8.4f} "
            f"lr={self.current_lr:.6f} "
            f"buffer={len(self.replay_buffer):05d}"
        )


def _validate_config(config: RandomDQNLightningConfig) -> None:
    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.replay_size < config.batch_size:
        raise ValueError("replay_size must be at least batch_size.")
    if config.replay_size < config.replay_warmup:
        raise ValueError("replay_size must be at least replay_warmup.")
    if config.target_sync_freq <= 0:
        raise ValueError("target_sync_freq must be positive.")


def _save_win_rate_plot(
    eval_history: list[dict[str, float]],
    train_win_rates: list[float],
    path: str | Path,
    title: str,
) -> None:
    """Save a combined training/evaluation win-rate figure."""

    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    if train_win_rates:
        plt.plot(train_win_rates, alpha=0.18, label="training win indicator")
        rolling_window = min(100, max(1, len(train_win_rates) // 5))
        if len(train_win_rates) >= rolling_window:
            kernel = np.ones(rolling_window) / rolling_window
            rolling = np.convolve(np.asarray(train_win_rates, dtype=np.float32), kernel, mode="valid")
            plt.plot(
                range(rolling_window - 1, rolling_window - 1 + len(rolling)),
                rolling,
                linewidth=2,
                label=f"{rolling_window}-episode training mean",
            )

    if eval_history:
        eval_episodes = [row["episode"] for row in eval_history]
        eval_win_rates = [row["win_rate"] for row in eval_history]
        plt.plot(eval_episodes, eval_win_rates, marker="o", linewidth=2, label="evaluation win rate")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("win rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=150)
    plt.close()


def train(config: RandomDQNLightningConfig) -> dict[str, Any]:
    """Train the random-mode DQN using official PyTorch Lightning."""

    _validate_config(config)
    set_seed(config.seed)
    ensure_dirs()

    agent = RandomDQNLightningModule(config)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu" if config.force_cpu else "auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=max(1, config.log_interval),
    )

    start_time = time.perf_counter()
    trainer.fit(agent)
    training_time = time.perf_counter() - start_time

    agent.sync_target_network()
    evaluation: EvaluationResult = evaluate_torch_q_network(
        agent.online_model,
        mode=config.mode,
        episodes=config.eval_episodes,
        max_steps=config.max_steps,
        device=str(agent.device),
    )
    random_baseline: EvaluationResult = evaluate_random_policy(
        mode=config.mode,
        episodes=config.random_baseline_episodes,
        max_steps=config.max_steps,
    )

    model_path = f"results/models/{config.output_prefix}.pt"
    torch.save(
        {
            "model_state_dict": agent.online_model.state_dict(),
            "target_model_state_dict": agent.target_model.state_dict(),
            "config": asdict(config),
            "evaluation": evaluation,
            "random_policy_baseline": random_baseline,
            "lightning_version": L.__version__,
        },
        model_path,
    )

    loss_plot = f"results/figures/{config.output_prefix}_loss.png"
    reward_plot = f"results/figures/{config.output_prefix}_reward.png"
    win_rate_plot = f"results/figures/{config.output_prefix}_win_rate.png"
    save_curve_plot(
        agent.losses,
        loss_plot,
        f"{config.output_prefix} Huber Loss",
        "loss",
        xlabel="Optimization step",
        rolling_window=min(100, max(1, len(agent.losses) // 10)) if agent.losses else None,
    )
    save_curve_plot(
        agent.episode_rewards,
        reward_plot,
        f"{config.output_prefix} Episode Reward",
        "episode reward",
        rolling_window=min(100, max(1, len(agent.episode_rewards) // 5)),
    )
    _save_win_rate_plot(
        eval_history=agent.eval_history,
        train_win_rates=agent.train_win_rates,
        path=win_rate_plot,
        title=f"{config.output_prefix} Win Rate",
    )

    summary = {
        "algorithm": "dueling_double_dqn_lightning",
        "framework": f"PyTorch Lightning {L.__version__} LightningModule + Trainer",
        "formal_lightning_trainer": True,
        "mode": config.mode,
        "episodes": config.episodes,
        "training_time_seconds": training_time,
        "global_steps": agent.env_steps,
        "optimization_steps": agent.optimization_steps,
        "final_epsilon": epsilon_by_episode(
            config.episodes - 1,
            config.epsilon_start,
            config.epsilon_final,
            config.epsilon_decay,
        ),
        "average_training_reward_last_100": float(np.mean(agent.episode_rewards[-100:])),
        "average_training_steps_last_100": float(np.mean(agent.episode_steps[-100:])),
        "training_win_rate_last_100": float(np.mean(agent.train_win_rates[-100:])),
        "training_tips": {
            "dueling_network": True,
            "double_dqn_target": True,
            "target_network": True,
            "experience_replay": True,
            "replay_warmup": config.replay_warmup,
            "huber_loss": config.use_huber_loss,
            "gradient_clip": config.gradient_clip,
            "epsilon_decay": {
                "start": config.epsilon_start,
                "final": config.epsilon_final,
                "decay": config.epsilon_decay,
            },
            "lr_scheduler": {
                "enabled": config.use_lr_scheduler,
                "step_size": config.scheduler_step_size,
                "gamma": config.scheduler_gamma,
                "final_lr": agent.learning_rates[-1] if agent.learning_rates else config.lr,
            },
            "periodic_evaluation": config.eval_interval > 0,
        },
        "evaluation": evaluation,
        "random_policy_baseline": random_baseline,
        "eval_history": agent.eval_history,
        "model_path": model_path,
        "loss_plot": loss_plot,
        "reward_plot": reward_plot,
        "win_rate_plot": win_rate_plot,
    }

    summary_path = Path(f"results/logs/{config.output_prefix}_summary.json")
    save_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=164)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-warmup", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=1000.0)
    parser.add_argument("--target-sync-freq", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--random-baseline-episodes", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-prefix", default="lightning_dqn_random")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--mse-loss", action="store_true", help="Use MSE instead of Huber loss.")
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--no-lr-scheduler", action="store_true", help="Disable StepLR scheduling.")
    parser.add_argument("--scheduler-step-size", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.95)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RandomDQNLightningConfig:
    gradient_clip = args.gradient_clip if args.gradient_clip > 0 else None
    return RandomDQNLightningConfig(
        output_prefix=args.output_prefix,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        replay_warmup=args.replay_warmup,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        target_sync_freq=args.target_sync_freq,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        random_baseline_episodes=args.random_baseline_episodes,
        log_interval=args.log_interval,
        seed=args.seed,
        force_cpu=args.cpu,
        use_huber_loss=not args.mse_loss,
        gradient_clip=gradient_clip,
        use_lr_scheduler=not args.no_lr_scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
    )


def main() -> None:
    summary = train(config_from_args(parse_args()))
    evaluation = summary["evaluation"]
    random_baseline = summary["random_policy_baseline"]
    print("\nTraining complete.")
    print(f"Evaluation win rate: {evaluation.win_rate:.2f}")
    print(f"Random-policy baseline win rate: {random_baseline.win_rate:.2f}")
    print(f"Saved results/logs/{Path(summary['model_path']).stem}_summary.json")


if __name__ == "__main__":
    main()
