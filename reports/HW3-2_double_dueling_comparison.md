# HW3-2 Double DQN 與 Dueling DQN 比較：Player Mode

## 1. 任務目標

HW3-2 要在 `player` mode 中實作並比較：

1. Double DQN
2. Dueling DQN

`player` mode 的 Goal、Pit、Wall 固定，但 Player 起點隨機，因此比 `static` mode 更需要泛化到不同初始位置。

## 2. 共用訓練設定

本專案將共用訓練流程整理在 `src/dqn_training.py`，兩個訓練入口分別為：

- `scripts/train_double_dqn_player.py`
- `scripts/train_dueling_dqn_player.py`

主要設定：

| 參數 | 預設值 |
| --- | ---: |
| mode | `player` |
| episodes | 1000 |
| max steps | 50 |
| gamma | 0.9 |
| learning rate | 1e-3 |
| replay size | 5000 |
| batch size | 64 |
| target sync freq | 100 steps |
| epsilon start / final | 1.0 / 0.05 |
| loss | Huber loss |
| gradient clipping | 5.0 |

## 3. Double DQN

一般 DQN 的 target 使用同一個 target network 直接取最大值：

$$
y = r + \gamma \max_{a'} Q_{target}(s', a')
$$

這可能產生 overestimation bias，因為 action selection 與 action evaluation 都受到最大化操作影響。

Double DQN 將 action selection 與 action evaluation 分開：

$$
a^* = \arg\max_{a'} Q_{online}(s', a')
$$

$$
y = r + \gamma Q_{target}(s', a^*)
$$

在本專案中：

- online network 選下一步 action。
- target network 評估該 action 的 Q-value。
- 實作位置：`src/dqn_training.py` 的 `_compute_targets()`。

## 4. Dueling DQN

Dueling DQN 將 Q-value 分解成 state value 與 action advantage：

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')
$$

直覺上：

- $V(s)$ 表示目前 state 本身好不好。
- $A(s,a)$ 表示在該 state 下某個 action 相對其他 action 的優勢。

當許多 action 的效果接近時，Dueling 架構可更有效地學習 state value。

在本專案中，Dueling network 定義於 `src/models.py` 的 `DuelingDQN`。

## 5. 執行方式

使用 `uv` 執行：

```bash
uv run python scripts/train_double_dqn_player.py
uv run python scripts/train_dueling_dqn_player.py
uv run python scripts/compare_hw3_2_player.py
```

快速測試可降低 episode 數：

```bash
uv run python scripts/train_double_dqn_player.py --episodes 100 --eval-episodes 50
uv run python scripts/train_dueling_dqn_player.py --episodes 100 --eval-episodes 50
uv run python scripts/compare_hw3_2_player.py
```

## 6. 輸出檔案

Double DQN：

- `results/models/double_dqn_player.pt`
- `results/logs/double_dqn_player_summary.json`
- `results/figures/double_dqn_player_loss.png`
- `results/figures/double_dqn_player_reward.png`
- `results/figures/double_dqn_player_win_rate.png`

Dueling DQN：

- `results/models/dueling_dqn_player.pt`
- `results/logs/dueling_dqn_player_summary.json`
- `results/figures/dueling_dqn_player_loss.png`
- `results/figures/dueling_dqn_player_reward.png`
- `results/figures/dueling_dqn_player_win_rate.png`

比較圖與表格：

- `results/logs/hw3_2_player_comparison.json`
- `results/figures/hw3_2_player_comparison.png`

## 7. 結果比較

正式訓練結果如下，數值來自：

- `results/logs/double_dqn_player_summary.json`
- `results/logs/dueling_dqn_player_summary.json`
- `results/logs/hw3_2_player_comparison.json`

| 指標 | Double DQN | Dueling DQN |
| --- | ---: | ---: |
| episodes | 1000 | 1000 |
| final epsilon | 0.0675 | 0.0675 |
| global steps | 7845 | 7634 |
| final win rate | 1.00 | 1.00 |
| loss rate | 0.00 | 0.00 |
| average reward | 6.62 | 6.74 |
| average steps | 4.38 | 4.26 |
| training win rate last 100 | 0.99 | 1.00 |
| training reward last 100 | 5.98 | 6.57 |
| training steps last 100 | 4.82 | 4.43 |
| training time | 18.80 秒 | 25.95 秒 |

## 8. 合理性檢查

這組結果合理。`player` mode 中 Goal、Pit、Wall 固定，只有 Player 起點隨機；合法起點扣除 Goal、Pit、Wall 與不合法位置後，主要需要學的是從不同起點繞過 Wall、避開 Pit 後抵達 Goal。

為確認結果是否過高，使用目前環境規則對所有合法 player 起點計算最短路徑。13 個合法起點的平均最短步數約為：

$$
4.3846
$$

若以最短路徑抵達 Goal，平均 reward 約為：

$$
10 - (4.3846 - 1) = 6.6154
$$

Double DQN 的 evaluation 結果為 average steps `4.38`、average reward `6.62`，幾乎等於理論最短路徑平均值；Dueling DQN 的 average steps `4.26`、average reward `6.74`，也在合理範圍內。Dueling DQN 略高可能來自 evaluation episodes 隨機抽樣下，較短路徑起點比例稍高；若提高 evaluation episodes，平均值應更接近長期平均。

因此，`final win rate = 1.00` 並不異常，表示兩種模型都已在 `player` mode 學到穩定策略。

## 9. 初步分析

相較於 HW3-1 的 `static` mode，`player` mode 中起點會變動，因此模型不能只記住單一路徑，而要學到從不同位置接近 Goal、避開 Pit 和 Wall 的策略。

Double DQN 的主要優勢是降低 Q-value overestimation，通常能讓 target 較穩定。Dueling DQN 的主要優勢是將 state value 與 action advantage 分開估計，對許多 action 差異不大的狀態可能更有效率。

本次結果中，兩者都達到 1.00 win rate。Double DQN 訓練時間較短，而 Dueling DQN 在最後 100 回合的 training reward、training win rate 與 evaluation average reward 略高。由於此環境仍然很小，兩者差異不應過度解讀；更困難的 `random` mode 才更適合觀察穩定性差異。

## 10. 下一步：HW3-3 Random Mode + PyTorch Lightning

下一階段應進入 HW3-3，將 DQN 訓練推進到 `random` mode，並轉成 PyTorch Lightning 或 Lightning-style 訓練流程。

建議下一步工作：

1. 建立 `scripts/train_lightning_dqn_random.py`。
2. 使用 `Gridworld(mode="random")`，讓 Player、Goal、Pit、Wall 全部隨機。
3. 沿用目前已驗證的 components：
   - replay buffer
   - target network
   - Double DQN target
   - DuelingDQN architecture，可作為穩定版本選項
4. 加入 training tips：
   - Huber loss
   - gradient clipping
   - epsilon decay
   - replay warmup
   - learning rate scheduler
   - 定期 evaluation
5. 輸出 random mode 的：
   - model checkpoint
   - loss curve
   - reward curve
   - win rate curve
   - summary JSON
6. 撰寫 `reports/HW3-3_random_mode_training_tips.md`，說明 random mode 為什麼更難，以及 training tips 如何幫助穩定訓練。
