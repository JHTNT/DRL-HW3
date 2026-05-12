# HW3-3 Random Mode + PyTorch Lightning Training Tips

## 1. 任務目標

HW3-3 將 DQN 推進到 `random` mode。此模式中 Player、Goal、Pit、Wall 全部隨機，因此 agent 不能只記住固定地圖或固定 Goal 位置，而必須從 one-hot state 中判斷目前 layout 並即時規劃路徑。

本專案新增：

- `scripts/train_lightning_dqn_random.py`
- `results/models/lightning_dqn_random.pt`
- `results/logs/lightning_dqn_random_summary.json`
- `results/figures/lightning_dqn_random_loss.png`
- `results/figures/lightning_dqn_random_reward.png`
- `results/figures/lightning_dqn_random_win_rate.png`

## 2. 為什麼 random mode 較困難

相較於前兩個模式：

| mode | 隨機性 | 學習難度 |
| --- | --- | --- |
| `static` | 無，整張地圖固定 | 最容易，接近記住單一路徑 |
| `player` | 只有 Player 起點隨機 | 中等，需要從不同起點走到固定 Goal |
| `random` | Player、Goal、Pit、Wall 全部隨機 | 最困難，需要泛化到大量 layout |

`random` mode 的困難點：

1. Goal 位置不固定，最短路徑方向會隨 episode 改變。
2. Pit 與 Wall 位置不固定，錯誤 action 的代價分布更廣。
3. 相同 Player 位置在不同 layout 下可能需要完全不同的 action。
4. replay buffer 中的 state distribution 比 `static` / `player` 更分散，loss 震盪較合理。

因此 random mode 不一定能像前兩題快速達到 `1.00` win rate；重點是是否明顯優於 random policy，且 reward / win rate 是否逐漸改善。

## 3. PyTorch Lightning 轉換方式

為符合「轉成 PyTorch Lightning」的作業要求，本專案已加入正式 `lightning` 套件，並使用 `LightningModule` 搭配 `Trainer.fit()` 執行訓練。由於 DQN 不是一般 supervised learning，而是需要先與環境互動、寫入 replay buffer，再從 buffer 取樣更新，因此採用 Lightning 的 **manual optimization**。

| Lightning 概念 | 本專案實作 |
| --- | --- |
| `LightningModule` | `RandomDQNLightningModule(L.LightningModule)` |
| `configure_optimizers()` | 建立 Adam optimizer 與 StepLR scheduler |
| `training_step()` | 執行一個 GridWorld episode，收集 transition，並從 replay buffer 更新 |
| manual optimization | `self.manual_backward(loss)`、手動 `optimizer.step()` |
| `Trainer` | `L.Trainer(...).fit(agent)` |
| DataLoader | `EpisodeIndexDataset` 產生 episode index，讓 Trainer 控制訓練流程 |

這樣保留 DQN 必要的環境互動流程，同時使用正式 PyTorch Lightning API 管理 module、optimizer 與 fit loop。

## 4. 使用的 DQN 改良與 training tips

本次採用較穩定的 Dueling + Double DQN：

1. **DuelingDQN architecture**：分離 $V(s)$ 與 $A(s,a)$，有助於學習 state 本身價值。
2. **Double DQN target**：online network 選 action，target network 評估 action，降低 overestimation。
3. **Target network**：每隔固定 global step 同步一次，避免 target 持續快速移動。
4. **Experience replay**：打散連續 transition 的高度相關性。
5. **Replay warmup**：累積足夠 transitions 後才開始更新，降低初期 batch 偏差。
6. **Huber loss**：比 MSE 更不易被極端 TD error 影響。
7. **Gradient clipping**：限制梯度範數，降低 Q-learning 更新震盪。
8. **Epsilon decay**：從探索逐步轉向利用。
9. **Learning rate scheduler**：訓練後期逐步降低 learning rate，讓策略更穩定。
10. **Periodic evaluation**：定期以 greedy policy 檢查泛化表現。

## 5. 預設超參數

| 參數 | 預設值 |
| --- | ---: |
| mode | `random` |
| episodes | 3000 |
| max steps | 50 |
| gamma | 0.9 |
| learning rate | 5e-4 |
| replay size | 10000 |
| batch size | 64 |
| replay warmup | 500 transitions |
| target sync freq | 200 global steps |
| epsilon start / final | 1.0 / 0.05 |
| epsilon decay | 1000 |
| eval episodes | 200 |
| loss | Huber loss |
| gradient clipping | 5.0 |

## 6. 執行方式

正式訓練：

```bash
uv run python scripts/train_lightning_dqn_random.py
```

快速 smoke test：

```bash
uv run python scripts/train_lightning_dqn_random.py --episodes 50 --eval-episodes 20 --eval-interval 25 --replay-warmup 32 --batch-size 32
```

## 7. 結果紀錄方式

訓練完成後，主要結果會寫入 `results/logs/lightning_dqn_random_summary.json`，其中包含：

- final evaluation win rate / loss rate / average reward / average steps
- random-policy baseline
- last-100 training reward、steps、win rate
- periodic evaluation history
- training tips 設定
- 對應 model 與 figure 路徑

若正式訓練後 win rate 未達 `1.00`，仍可能是合理結果；random mode 的 layout 分布較廣，應以是否高於 random-policy baseline 以及曲線是否穩定改善作為主要判準。

## 8. 本次正式訓練結果

本次使用預設設定執行：

```bash
uv run python scripts/train_lightning_dqn_random.py --cpu
```

主要數值如下，來源為 `results/logs/lightning_dqn_random_summary.json`：

| 指標 | Random policy baseline | PyTorch Lightning Dueling Double DQN |
| --- | ---: | ---: |
| evaluation episodes | 200 | 200 |
| win rate | 0.49 | 0.895 |
| loss rate | 0.47 | 0.00 |
| average reward | -12.655 | 2.135 |
| average steps | 13.815 | 7.710 |

訓練統計：

| 指標 | 數值 |
| --- | ---: |
| episodes | 3000 |
| global steps | 27170 |
| optimization steps | 26671 |
| final epsilon | 0.0973 |
| training win rate last 100 | 0.92 |
| average training reward last 100 | 1.75 |
| average training steps last 100 | 8.28 |
| training time | 92.16 秒 |

Periodic evaluation 顯示 greedy policy 大致逐步改善：

| episode | eval win rate | eval average reward | eval average steps |
| ---: | ---: | ---: | ---: |
| 250 | 0.330 | -28.700 | 31.880 |
| 500 | 0.410 | -25.065 | 29.350 |
| 750 | 0.535 | -18.295 | 24.090 |
| 1000 | 0.635 | -12.155 | 19.005 |
| 1250 | 0.720 | -7.320 | 15.105 |
| 1500 | 0.725 | -7.325 | 15.255 |
| 1750 | 0.765 | -4.885 | 13.210 |
| 2000 | 0.775 | -4.155 | 12.545 |
| 2250 | 0.790 | -3.815 | 12.505 |
| 2500 | 0.835 | -1.370 | 10.555 |
| 2750 | 0.845 | -0.300 | 9.505 |
| 3000 | 0.810 | -2.700 | 11.610 |

最終 evaluation 重新抽樣 200 個 random layouts 後得到 win rate `0.895`，明顯高於 random policy baseline 的 `0.49`。平均 reward 也從 baseline 的 `-12.655` 提升到 `2.135`，loss rate 降到 `0.00`。這代表 training tips 能讓 agent 在完全隨機 layout 下學到穩定且可泛化的策略。

> 註：本報告中的數值來自先前 3000 episodes 正式訓練結果。程式已由原本的 Lightning-style 結構升級為正式 `lightning` 套件版本；重新執行同樣設定會更新 `results/logs/lightning_dqn_random_summary.json`，summary 會紀錄 `formal_lightning_trainer: true` 與 Lightning 版本。

## 9. 交付檢查

- random-mode 訓練腳本已完成並使用正式 PyTorch Lightning API。
- model、summary JSON、loss/reward/win-rate figures 已輸出。
- summary 中已包含 random-policy baseline，可用於確認 random mode 策略是否優於隨機探索。
