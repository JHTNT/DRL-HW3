# HW3-1 Understanding Report：Naive DQN for Static GridWorld

## 1. 任務環境

本作業使用 DRL in Action Chapter 3 的 4x4 GridWorld。專案中保留原始參考檔於 `reference/`，並將可重複使用的環境整理到 `src/gridworld.py` 與 `src/gridboard.py`。

### State

`GridBoard.render_np()` 會把棋盤轉成 one-hot layer：

```text
(num_pieces, height, width) = (4, 4, 4)
```

四個 layer 分別對應：

1. Player
2. Goal
3. Pit
4. Wall

DQN 訓練時將 state 攤平成：

```text
(1, 64)
```

### Action

動作空間共有四個離散動作：

| action index |   action    |
| :----------: | :---------: |
|      0       |  up (`u`)   |
|      1       | down (`d`)  |
|      2       | left (`l`)  |
|      3       | right (`r`) |

### Reward

|      狀態      | reward |
| :------------: | :----: |
| Player 到 Goal |  +10   |
| Player 到 Pit  |  -10   |
|    其他移動    |   -1   |

`static` mode 中 Player、Goal、Pit、Wall 的位置固定，因此是最容易開始測試 DQN 是否能學到穩定策略的設定。

## 2. DQN Network Architecture

本次實作在 `src/models.py` 中使用簡單 MLP：

```text
Linear(64, 164)
ReLU
Linear(164, 164)
ReLU
Linear(164, 4)
```

輸入是 flattened state，輸出是四個 action 的 Q-value：

```text
Q(s, up), Q(s, down), Q(s, left), Q(s, right)
```

模型會選擇 Q-value 最大的 action 作為 greedy action。

## 3. Naive DQN Target

Naive DQN 每一步直接用目前網路估計下一狀態最大 Q-value：

$$
y = r + \gamma \max_{a'} Q(s', a')
$$

如果 episode 已結束，則不再 bootstrap：

$$
y = r
$$

本專案中的 naive 版本在 `scripts/train_naive_dqn_static.py`。

## 4. Epsilon-Greedy 的用途

訓練時使用 epsilon-greedy 平衡 exploration 與 exploitation：

- 以 $\epsilon$ 機率隨機選 action。
- 以 $1 - \epsilon$ 機率選目前模型認為 Q-value 最大的 action。

訓練初期需要較高 exploration 避免模型只嘗試少數路徑；訓練後期降低 epsilon，讓模型逐漸使用學到的策略。

## 5. Experience Replay Buffer 的用途

Experience replay 將 transition 存入 buffer：

```text
(state, action, reward, next_state, done)
```

每次訓練時從 buffer 隨機抽 mini-batch。優點：

1. 降低連續樣本之間的高度相關性。
2. 讓過去經驗可以被重複利用。
3. mini-batch 更新通常比 single-step update 穩定。

本專案中的 replay buffer 在 `src/replay_buffer.py`，訓練腳本是 `scripts/train_dqn_replay_static.py`。

## 6. 執行方式

建議使用 `uv`：

```bash
uv run python scripts/train_naive_dqn_static.py
uv run python scripts/train_dqn_replay_static.py
```

快速測試可降低 episode 數：

```bash
uv run python scripts/train_dqn_replay_static.py --episodes 100 --eval-episodes 50
```

## 7. 輸出檔案

訓練後會產生：

- `results/models/naive_dqn_static.pt`
- `results/models/dqn_replay_static.pt`
- `results/logs/naive_dqn_static_summary.json`
- `results/logs/dqn_replay_static_summary.json`
- `results/figures/naive_dqn_static_loss.png`
- `results/figures/naive_dqn_static_reward.png`
- `results/figures/dqn_replay_static_loss.png`
- `results/figures/dqn_replay_static_reward.png`

## 8. 初步觀察

Static mode 的最佳路徑固定，因此 DQN 應能比 random policy 明顯更穩定地到達 Goal。實際以 GPU 執行 500 episodes 後，naive DQN 與 replay DQN 都很快完成訓練，evaluation win rate 皆達到 `1.00`。

|        方法         | episodes | eval win rate | eval loss rate | eval average reward | eval average steps | training time |
| :-----------------: | :------: | :-----------: | :------------: | :-----------------: | :----------------: | :-----------: |
|      Naive DQN      |   500    |     1.00      |      0.00      |        4.00         |        7.00        |   10.81 秒    |
| DQN + Replay Buffer |   500    |     1.00      |      0.00      |        4.00         |        7.00        |   12.05 秒    |

這個結果合理，因為 `static` mode 中 Player、Goal、Pit、Wall 位置皆固定，狀態與最佳路徑都相對簡單，DQN 很容易記住穩定策略。平均 reward 為 `4.00`，對應到約 7 steps 抵達 Goal：前 6 步各得到 `-1`，最後到達 Goal 得到 `+10`，總和為 `4`。

若訓練結果不穩定，常見調整方向包含：

1. 增加 episode 數。
2. 降低 learning rate。
3. 放慢 epsilon decay。
4. 使用 replay buffer。
5. 進一步加入 target network，作為後續 Double DQN / Dueling DQN 的基礎。
