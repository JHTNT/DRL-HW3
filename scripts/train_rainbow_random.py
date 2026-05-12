"""HW3-4 bonus: train Rainbow DQN on random-mode GridWorld."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from src.gridworld import ACTIONS, Gridworld
from src.rainbow import C51RainbowDQN, NStepTransitionBuffer, PrioritizedReplayBuffer, RainbowDQN
from src.train_utils import (
    EvaluationResult,
    ensure_dirs,
    epsilon_by_episode,
    evaluate_random_policy,
    evaluate_torch_q_network,
    get_torch_device,
    save_curve_plot,
    save_json,
    set_seed,
)


@dataclass
class RainbowTrainingConfig:
    """Configuration for Rainbow DQN training."""

    mode: str = "random"
    output_prefix: str = "rainbow_random"
    episodes: int = 3000
    max_steps: int = 50
    gamma: float = 0.9
    lr: float = 5e-4
    hidden_dim: int = 164
    replay_size: int = 10000
    batch_size: int = 64
    replay_warmup: int = 500
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_final: float = 1.0
    epsilon_start: float = 0.2
    epsilon_final: float = 0.01
    epsilon_decay: float = 1000.0
    target_sync_freq: int = 200
    eval_episodes: int = 200
    eval_interval: int = 250
    random_baseline_episodes: int = 200
    log_interval: int = 100
    seed: int = 13
    force_cpu: bool = False
    use_noisy: bool = True
    noisy_sigma: float = 0.5
    use_distributional: bool = True
    atom_size: int = 51
    v_min: float = -50.0
    v_max: float = 10.0
    gradient_clip: float | None = 5.0
    priority_epsilon: float = 1e-5
    use_lr_scheduler: bool = True
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.95


class RainbowAgent(nn.Module):
    """Rainbow agent with PER, n-step returns, NoisyNet, C51, and Double DQN."""

    def __init__(self, config: RainbowTrainingConfig):
        super().__init__()
        self.config = config
        self.online_model = self._build_model(config)
        self.target_model = self._build_model(config)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_size,
            alpha=config.per_alpha,
            priority_epsilon=config.priority_epsilon,
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.sync_target_network()

    @staticmethod
    def _build_model(config: RainbowTrainingConfig) -> nn.Module:
        if config.use_distributional:
            return C51RainbowDQN(
                state_dim=64,
                action_dim=len(ACTIONS),
                hidden_dim=config.hidden_dim,
                atom_size=config.atom_size,
                v_min=config.v_min,
                v_max=config.v_max,
                use_noisy=config.use_noisy,
                noisy_sigma=config.noisy_sigma,
            )
        return RainbowDQN(
            state_dim=64,
            action_dim=len(ACTIONS),
            hidden_dim=config.hidden_dim,
            use_noisy=config.use_noisy,
            noisy_sigma=config.noisy_sigma,
        )

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, Any | None]:
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.config.lr)
        self.scheduler = None
        if self.config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        return self.optimizer, self.scheduler

    @property
    def current_lr(self) -> float:
        if self.optimizer is None:
            return self.config.lr
        return float(self.optimizer.param_groups[0]["lr"])

    def sync_target_network(self) -> None:
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    def reset_noise(self) -> None:
        self.online_model.reset_noise()
        self.target_model.reset_noise()

    def select_action(self, state: np.ndarray, epsilon: float, device: torch.device) -> int:
        if random.random() < epsilon:
            return int(np.random.randint(0, len(ACTIONS)))

        self.online_model.reset_noise()
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            return int(torch.argmax(self.online_model(state_t), dim=1).item())

    def training_step(self, beta: float, device: torch.device) -> dict[str, float] | None:
        if self.optimizer is None:
            raise RuntimeError("configure_optimizers() must be called before training_step().")

        min_replay_size = max(self.config.batch_size, self.config.replay_warmup)
        if len(self.replay_buffer) < min_replay_size:
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(
            self.config.batch_size,
            beta=beta,
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        self.online_model.reset_noise()
        self.target_model.reset_noise()

        if self.config.use_distributional:
            action_indices = actions_t.unsqueeze(-1).expand(-1, 1, self.config.atom_size)
            chosen_dist = self.online_model.dist(states_t).gather(1, action_indices).squeeze(1)
            target_dist = self._project_distribution(rewards_t, next_states_t, dones_t, device)
            elementwise_loss = -(target_dist * chosen_dist.log()).sum(dim=1)
            td_errors = elementwise_loss.detach().cpu().numpy()
        else:
            q_values = self.online_model(states_t).gather(1, actions_t).squeeze(1)
            with torch.no_grad():
                next_actions = self.online_model(next_states_t).argmax(dim=1, keepdim=True)
                next_q_values = self.target_model(next_states_t).gather(1, next_actions).squeeze(1)
                n_step_gamma = self.config.gamma**self.config.n_step
                targets = rewards_t + n_step_gamma * next_q_values * (1.0 - dones_t)

            elementwise_loss = self.loss_fn(q_values, targets)
            td_errors = torch.abs(targets - q_values).detach().cpu().numpy()

        loss = (elementwise_loss * weights_t).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.replay_buffer.update_priorities(indices, td_errors + self.config.priority_epsilon)
        return {
            "loss": float(loss.item()),
            "td_error": float(np.mean(td_errors)),
            "lr": self.current_lr,
        }

    def _project_distribution(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Project n-step Double DQN target distributions onto the C51 support."""

        assert isinstance(self.online_model, C51RainbowDQN)
        assert isinstance(self.target_model, C51RainbowDQN)

        batch_size = rewards.shape[0]
        support = self.online_model.support.to(device)
        delta_z = self.online_model.delta_z

        with torch.no_grad():
            next_actions = self.online_model(next_states).argmax(dim=1)
            next_dist = self.target_model.dist(next_states)
            next_dist = next_dist[torch.arange(batch_size, device=device), next_actions]

            n_step_gamma = self.config.gamma**self.config.n_step
            target_support = rewards.unsqueeze(1) + n_step_gamma * support.unsqueeze(0) * (1.0 - dones.unsqueeze(1))
            target_support = target_support.clamp(self.config.v_min, self.config.v_max)
            b = (target_support - self.config.v_min) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            projection = torch.zeros(batch_size, self.config.atom_size, device=device)
            offset = (
                torch.arange(batch_size, device=device).unsqueeze(1) * self.config.atom_size
            )

            lower_weight = (upper.float() - b)
            upper_weight = (b - lower.float())
            equal_mask = upper == lower
            lower_weight[equal_mask] = 1.0
            upper_weight[equal_mask] = 0.0

            projection.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (next_dist * lower_weight).view(-1),
            )
            projection.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (next_dist * upper_weight).view(-1),
            )

        return projection


def _validate_config(config: RainbowTrainingConfig) -> None:
    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.replay_size < config.batch_size:
        raise ValueError("replay_size must be at least batch_size.")
    if config.replay_size < config.replay_warmup:
        raise ValueError("replay_size must be at least replay_warmup.")
    if config.n_step <= 0:
        raise ValueError("n_step must be positive.")
    if config.target_sync_freq <= 0:
        raise ValueError("target_sync_freq must be positive.")
    if config.use_distributional:
        if config.atom_size <= 1:
            raise ValueError("atom_size must be greater than 1 when distributional DQN is enabled.")
        if config.v_min >= config.v_max:
            raise ValueError("v_min must be smaller than v_max.")


def _beta_by_episode(config: RainbowTrainingConfig, episode: int) -> float:
    if config.episodes <= 1:
        return config.per_beta_final
    progress = episode / (config.episodes - 1)
    return float(config.per_beta_start + progress * (config.per_beta_final - config.per_beta_start))


def _save_win_rate_plot(
    eval_history: list[dict[str, float]],
    train_win_rates: list[float],
    path: str | Path,
    title: str,
) -> None:
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


def _push_n_step_transition(
    agent: RainbowAgent,
    n_step_buffer: NStepTransitionBuffer,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
) -> None:
    transition = n_step_buffer.push(state, action, reward, next_state, done)
    if transition is not None:
        agent.replay_buffer.push(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        )

    if done:
        for remaining_transition in n_step_buffer.flush():
            agent.replay_buffer.push(
                remaining_transition.state,
                remaining_transition.action,
                remaining_transition.reward,
                remaining_transition.next_state,
                remaining_transition.done,
            )


def _flush_n_step_transitions(agent: RainbowAgent, n_step_buffer: NStepTransitionBuffer) -> None:
    """Push remaining shorter-horizon transitions after an episode timeout."""

    for remaining_transition in n_step_buffer.flush():
        agent.replay_buffer.push(
            remaining_transition.state,
            remaining_transition.action,
            remaining_transition.reward,
            remaining_transition.next_state,
            remaining_transition.done,
        )


def train(config: RainbowTrainingConfig) -> dict[str, Any]:
    """Train Rainbow DQN and save model, curves, and summary."""

    _validate_config(config)
    set_seed(config.seed)
    ensure_dirs()
    device = get_torch_device(force_cpu=config.force_cpu)

    agent = RainbowAgent(config).to(device)
    agent.configure_optimizers()
    agent.online_model.train()
    agent.target_model.eval()

    losses: list[float] = []
    td_errors: list[float] = []
    learning_rates: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    train_win_rates: list[float] = []
    eval_history: list[dict[str, float]] = []
    global_step = 0
    optimization_steps = 0
    start_time = time.perf_counter()

    for episode in range(config.episodes):
        env = Gridworld(size=4, mode=config.mode)
        n_step_buffer = NStepTransitionBuffer(config.n_step, config.gamma)
        total_reward = 0.0
        episode_step = 0
        episode_done = False
        won = False
        epsilon = 0.0
        if not config.use_noisy:
            epsilon = epsilon_by_episode(
                episode,
                epsilon_start=config.epsilon_start,
                epsilon_final=config.epsilon_final,
                epsilon_decay=config.epsilon_decay,
            )
        beta = _beta_by_episode(config, episode)

        for step in range(1, config.max_steps + 1):
            episode_step = step
            global_step += 1
            state = env.state()
            action = agent.select_action(state, epsilon, device)

            env.make_move(action)
            reward = float(env.reward())
            next_state = env.state()
            done = reward in {-10.0, 10.0}
            episode_done = done
            total_reward += reward
            won = reward == 10.0

            _push_n_step_transition(agent, n_step_buffer, state, action, reward, next_state, done)
            step_result = agent.training_step(beta, device)
            if step_result is not None:
                losses.append(step_result["loss"])
                td_errors.append(step_result["td_error"])
                learning_rates.append(step_result["lr"])
                optimization_steps += 1

            if global_step % config.target_sync_freq == 0:
                agent.sync_target_network()

            if done:
                break

        if not episode_done:
            _flush_n_step_transitions(agent, n_step_buffer)

        episode_rewards.append(total_reward)
        episode_steps.append(episode_step)
        train_win_rates.append(1.0 if won else 0.0)

        should_eval = config.eval_interval > 0 and (episode + 1) % config.eval_interval == 0
        if should_eval:
            eval_result = evaluate_torch_q_network(
                agent.online_model,
                mode=config.mode,
                episodes=config.eval_episodes,
                max_steps=config.max_steps,
                device=str(device),
            )
            eval_history.append(
                {
                    "episode": float(episode + 1),
                    "win_rate": eval_result.win_rate,
                    "loss_rate": eval_result.loss_rate,
                    "average_reward": eval_result.average_reward,
                    "average_steps": eval_result.average_steps,
                }
            )
            agent.online_model.train()
            agent.target_model.eval()

        if (episode + 1) % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval :]
            recent_wins = train_win_rates[-config.log_interval :]
            recent_loss = float(np.mean(losses[-config.log_interval :])) if losses else float("nan")
            recent_td_error = float(np.mean(td_errors[-config.log_interval :])) if td_errors else float("nan")
            print(
                f"rainbow random episode={episode + 1:04d} "
                f"epsilon={epsilon:.3f} beta={beta:.3f} "
                f"avg_reward={np.mean(recent_rewards):6.2f} "
                f"win_rate={np.mean(recent_wins):.2f} "
                f"avg_loss={recent_loss:8.4f} "
                f"avg_td={recent_td_error:8.4f} "
                f"lr={agent.current_lr:.6f} "
                f"buffer={len(agent.replay_buffer):05d}"
            )

    agent.sync_target_network()
    training_time = time.perf_counter() - start_time
    evaluation: EvaluationResult = evaluate_torch_q_network(
        agent.online_model,
        mode=config.mode,
        episodes=config.eval_episodes,
        max_steps=config.max_steps,
        device=str(device),
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
        },
        model_path,
    )

    loss_plot = f"results/figures/{config.output_prefix}_loss.png"
    reward_plot = f"results/figures/{config.output_prefix}_reward.png"
    win_rate_plot = f"results/figures/{config.output_prefix}_win_rate.png"
    save_curve_plot(
        losses,
        loss_plot,
        f"{config.output_prefix} {'C51 Cross-Entropy' if config.use_distributional else 'Weighted Huber'} Loss",
        "loss",
        xlabel="Optimization step",
        rolling_window=min(100, max(1, len(losses) // 10)) if losses else None,
    )
    save_curve_plot(
        episode_rewards,
        reward_plot,
        f"{config.output_prefix} Episode Reward",
        "episode reward",
        rolling_window=min(100, max(1, len(episode_rewards) // 5)),
    )
    _save_win_rate_plot(
        eval_history=eval_history,
        train_win_rates=train_win_rates,
        path=win_rate_plot,
        title=f"{config.output_prefix} Win Rate",
    )

    summary = {
        "algorithm": "rainbow_dqn" if config.use_distributional else "rainbow_style_dqn",
        "mode": config.mode,
        "episodes": config.episodes,
        "training_time_seconds": training_time,
        "global_steps": global_step,
        "optimization_steps": optimization_steps,
        "final_epsilon": 0.0
        if config.use_noisy
        else epsilon_by_episode(
            config.episodes - 1,
            config.epsilon_start,
            config.epsilon_final,
            config.epsilon_decay,
        ),
        "final_beta": _beta_by_episode(config, config.episodes - 1),
        "average_training_reward_last_100": float(np.mean(episode_rewards[-100:])),
        "average_training_steps_last_100": float(np.mean(episode_steps[-100:])),
        "training_win_rate_last_100": float(np.mean(train_win_rates[-100:])),
        "rainbow_components": {
            "double_dqn_target": True,
            "dueling_network": True,
            "prioritized_replay": True,
            "n_step_return": config.n_step,
            "noisy_network": config.use_noisy,
            "distributional_dqn": config.use_distributional,
            "c51_atoms": config.atom_size if config.use_distributional else None,
            "v_min": config.v_min if config.use_distributional else None,
            "v_max": config.v_max if config.use_distributional else None,
        },
        "training_tips": {
            "target_network": True,
            "huber_loss": True,
            "gradient_clip": config.gradient_clip,
            "replay_warmup": config.replay_warmup,
            "lr_scheduler": {
                "enabled": config.use_lr_scheduler,
                "step_size": config.scheduler_step_size,
                "gamma": config.scheduler_gamma,
                "final_lr": learning_rates[-1] if learning_rates else config.lr,
            },
            "per_alpha": config.per_alpha,
            "per_beta_start": config.per_beta_start,
            "per_beta_final": config.per_beta_final,
        },
        "evaluation": evaluation,
        "random_policy_baseline": random_baseline,
        "eval_history": eval_history,
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
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-final", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-final", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=1000.0)
    parser.add_argument("--target-sync-freq", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--random-baseline-episodes", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-prefix", default="rainbow_random")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--no-noisy", action="store_true", help="Disable NoisyNet and use epsilon-greedy.")
    parser.add_argument("--noisy-sigma", type=float, default=0.5)
    parser.add_argument("--no-distributional", action="store_true", help="Disable C51 and use scalar Q targets.")
    parser.add_argument("--atom-size", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-50.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--priority-epsilon", type=float, default=1e-5)
    parser.add_argument("--no-lr-scheduler", action="store_true", help="Disable StepLR scheduling.")
    parser.add_argument("--scheduler-step-size", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.95)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RainbowTrainingConfig:
    gradient_clip = args.gradient_clip if args.gradient_clip > 0 else None
    return RainbowTrainingConfig(
        output_prefix=args.output_prefix,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        replay_warmup=args.replay_warmup,
        n_step=args.n_step,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_final=args.per_beta_final,
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
        use_noisy=not args.no_noisy,
        noisy_sigma=args.noisy_sigma,
        use_distributional=not args.no_distributional,
        atom_size=args.atom_size,
        v_min=args.v_min,
        v_max=args.v_max,
        gradient_clip=gradient_clip,
        priority_epsilon=args.priority_epsilon,
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