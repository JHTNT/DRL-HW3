# HW3-4 Rainbow DQN 加分題：Random Mode GridWorld

## 1. 任務目標

HW3-4 加分題要求使用 Rainbow DQN 解 `random` mode GridWorld。本專案已實作 Rainbow DQN 的主要 components：Double DQN、Dueling DQN、Prioritized Experience Replay、Multi-step return、NoisyNet，以及 C51 Distributional DQN。

新增檔案：

- `src/rainbow.py`
- `scripts/train_rainbow_random.py`
- `reports/HW3-4_rainbow_bonus.md`

預期輸出：

- `results/models/rainbow_random.pt`
- `results/logs/rainbow_random_summary.json`
- `results/figures/rainbow_random_loss.png`
- `results/figures/rainbow_random_reward.png`
- `results/figures/rainbow_random_win_rate.png`

## 2. 本專案實作的 Rainbow components

| Component | 是否實作 | 說明 |
| --- | --- | --- |
| Double DQN | 是 | online network 選下一步 action，target network 評估該 action |
| Dueling DQN | 是 | `C51RainbowDQN` 使用 value stream 與 advantage stream |
| Prioritized Experience Replay | 是 | `PrioritizedReplayBuffer` 依 TD error priority 抽樣 |
| Multi-step return | 是 | `NStepTransitionBuffer` 預設使用 3-step return |
| NoisyNet | 是 | `NoisyLinear` 用於 value / advantage streams 的探索 |
| Distributional DQN / C51 | 是 | `C51RainbowDQN` 輸出每個 action 的 categorical value distribution |

因此目前版本已從原本的 **Rainbow-style DQN** 補強為包含 C51 的 Rainbow DQN。若要做 ablation，可用 `--no-distributional` 關閉 C51，退回 scalar Q-target 版本。

## 3. 核心實作說明

### 3.1 NoisyNet

`src/rainbow.py` 的 `NoisyLinear` 實作 factorized Gaussian noise。訓練期間 network weight 會加入可學習的 noise scale，讓 agent 不只依賴 epsilon-greedy，而能透過參數噪聲探索不同策略。

在 `scripts/train_rainbow_random.py` 預設 `use_noisy=True`，因此 `epsilon=0`；若用 `--no-noisy` 則會改用 epsilon decay。

### 3.2 Dueling network

`C51RainbowDQN` 使用 dueling distributional head。每個 action 不只輸出單一 Q-value，而是輸出 `atom_size=51` 個 atoms 上的機率分布。其期望值仍可作為 greedy action 的 Q-value：

$$
Q(s,a)=\sum_i z_i p_i(s,a)
$$

Dueling aggregation 使用：

$$
Q(s,a)=V(s)+A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a')
$$

在 distributional 版本中，value stream 與 advantage stream 輸出的 logits 會 reshape 成 `(batch, action, atom)` 後再做 softmax 得到 categorical distribution。這讓模型能同時保留 Dueling 的 state/action 分解與 C51 的價值分布估計。

### 3.3 Prioritized replay

`PrioritizedReplayBuffer` 使用 proportional prioritization：

$$
P(i)=\frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

並使用 importance-sampling weight 修正抽樣偏差：

$$
w_i=(N \cdot P(i))^{-\beta}
$$

其中 `beta` 會從 `0.4` 線性增加到 `1.0`。

### 3.4 N-step return

`NStepTransitionBuffer` 將連續 transition 合併為 n-step target，預設 `n_step=3`：

$$
R_t^{(n)}=r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}
$$

若關閉 C51 時，scalar Q 版本的訓練 target 使用：

$$
y=R_t^{(n)}+\gamma^n Q_{target}(s_{t+n}, \arg\max_a Q_{online}(s_{t+n},a))
$$

若 episode 提前到達 Goal 或 Pit，則 bootstrap 項會被移除。

### 3.5 C51 Distributional DQN

C51 不直接回歸 scalar target，而是在固定 support 上學習 value distribution。本專案預設：

```text
atom_size = 51
v_min = -50
v_max = 10
```

Double DQN 先用 online network 選下一狀態 action：

$$
a^*=\arg\max_a Q_{online}(s_{t+n},a)
$$

再用 target network 取得該 action 的 distribution，並將 n-step target distribution 投影回固定 support：

$$
Tz_j=\text{clip}(R_t^{(n)}+\gamma^n z_j, V_{min}, V_{max})
$$

訓練 loss 使用 target distribution 與目前 distribution 的 cross entropy，PER priority 則使用每筆樣本的 distributional loss 作為 TD-error proxy。

## 4. 預設超參數

| 參數 | 預設值 |
| --- | ---: |
| mode | `random` |
| episodes | 3000 |
| max steps | 50 |
| gamma | 0.9 |
| learning rate | 5e-4 |
| replay size | 10000 |
| batch size | 64 |
| replay warmup | 500 |
| n-step | 3 |
| C51 atoms | 51 |
| C51 support | [-50, 10] |
| PER alpha | 0.6 |
| PER beta start / final | 0.4 / 1.0 |
| target sync freq | 200 steps |
| NoisyNet | enabled |
| Huber loss | enabled |
| gradient clipping | 5.0 |
| LR scheduler | StepLR |

## 5. 執行方式

正式訓練：

```bash
uv run python scripts/train_rainbow_random.py
```

快速 smoke test：

```bash
uv run python scripts/train_rainbow_random.py --episodes 50 --eval-episodes 20 --eval-interval 25 --replay-warmup 32 --batch-size 32
```

若要關閉 NoisyNet、改用 epsilon-greedy：

```bash
uv run python scripts/train_rainbow_random.py --no-noisy
```

若要關閉 C51、退回原本 scalar Q-target 的 Rainbow-style ablation：

```bash
uv run python scripts/train_rainbow_random.py --no-distributional
```

## 6. 與 HW3-3 的比較方式

HW3-3 的 PyTorch Lightning Dueling Double DQN 已在 `random` mode 取得：

| 指標 | HW3-3 PyTorch Lightning DQN |
| --- | ---: |
| evaluation win rate | 0.895 |
| loss rate | 0.00 |
| average reward | 2.135 |
| average steps | 7.710 |
| random-policy baseline win rate | 0.49 |

HW3-4 完成訓練後，可將 `results/logs/rainbow_random_summary.json` 的 evaluation 欄位與上表比較。由於 random mode 每次 evaluation 會抽樣不同 layouts，單次結果可能有波動；應主要觀察：

1. 是否高於 random policy baseline。
2. win-rate curve 是否隨訓練上升。
3. average reward 是否由負轉正或明顯改善。
4. loss 與 TD error 是否沒有發散。

## 7. 交付檢查

- Rainbow components 已集中於 `src/rainbow.py`。
- 訓練入口為 `scripts/train_rainbow_random.py`。
- Summary JSON 會紀錄已實作 components，包含 Double DQN、Dueling、PER、n-step、NoisyNet，以及 C51 Distributional DQN。
