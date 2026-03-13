# Timeframe-Aware Representation Learning for Multi-Resolution Deep Q-Trading

## Abstract

Deep reinforcement learning has shown promise for algorithmic trading, particularly when decisions are made from market signals observed across multiple temporal horizons. However, many multi-resolution trading architectures still rely on flat feature concatenation followed by uniform dense processing, which obscures the relative importance of hourly, daily, and weekly information and limits interpretability. In this paper, we propose a modular timeframe-aware attention framework for multi-resolution Deep Q-Trading. The framework introduces three complementary attention mechanisms that operate at different levels of granularity: Global Feature Attention (GFA), which reweights the full input representation; Local Feature Attention (LFA), which emphasizes salient features within each temporal segment; and Time-Frame Attention (TFA), which explicitly models the relative contribution of entire temporal resolutions through a gating mechanism. Unlike feature-only reweighting schemes, TFA is designed to capture hierarchical market dynamics by learning how short-, medium-, and longer-horizon signals should be prioritized under changing market conditions. Importantly, the proposed modules preserve the original optimization and training pipeline, enabling a controlled analysis of representational effects independent of algorithmic changes. Experiments on DAX, S&P 500, and MSFT under walk-forward evaluation show that timeframe-aware weighting consistently improves robustness and profitability, with TFA delivering the strongest and most stable performance across datasets and ensemble agreement thresholds. These findings highlight the importance of modeling timeframe-specific information in multi-resolution reinforcement-learning-based trading.

---

## Description

### Core files

-   **main.py**: Entry point of the application
-   **expert/deepQTrading.py**: Organizes data, sets up and trains the DQN agents with walk-forward evaluation
-   **expert/attention_network.py**: Attention model definitions and model builder
-   **environments/spEnv.py**: OpenAI Gym-compatible trading environment
-   **environments/mergedDataStructure.py**: Multi-timeframe data structure (hourly / daily / weekly)
-   **utils/market_config.py**: Market configurations (DAX, S&P 500, MSFT)
-   **utils/callback.py**: Callbacks for logging, Q-value tracking, and attention weight recording
-   **evaluation/evaluation.py**: Post-training evaluation and result plotting

### Attention modules (`expert/attention_network.py`)

-   **GlobalFeatureAttention (GFA)**: applies a learned attention weight over the full concatenated feature vector (68-dim), reweighting all features jointly
-   **LocalFeatureAttention (LFA)**: applies independent attention weights within each temporal segment — hourly (40-dim), daily (20-dim), and weekly (8-dim) — before concatenating the results
-   **TimeFrameAttention (TFA)**: computes a scalar importance score per temporal resolution, produces a 3-way softmax over timeframes, scales each segment by its gate weight, then applies a sigmoid gate over the weighted representation

---

## Requirements

-   Python 3.6.8
-   TensorFlow 1.14.0: `pip install tensorflow==1.14.0`
-   Keras 2.3.1: `pip install keras==2.3.1`
-   Keras-RL 0.4.2: `pip install keras-rl==0.4.2`
-   OpenAI Gym 0.26.2: `pip install gym==0.26.2`
-   Pandas 0.25.3: `pip install pandas==0.25.3`
-   NumPy 1.19.5: `pip install numpy==1.19.5`

---

## Datasets

Place raw market data CSV files in `./datasets/`. Expected naming convention:

```
datasets/
  daxHour.csv    daxDay.csv    daxWeek.csv
  sp500Hour.csv  sp500Day.csv  sp500Week.csv
  msftHour.csv   msftDay.csv   msftWeek.csv
```

Each file should contain `Date`, `Time`, `Open`, `High`, `Low`, `Close`, `Volume` columns. Walk-forward evaluation uses a 5-year training window, 6-month validation, and 6-month test window, sliding by 6 months per walk (8 walks total, 2007–2017).

---

## Usage

Run the experiment for a given market and model variant:

```bash
python -W ignore main.py <market> <model>
```

**`<market>`** — one of: `dax`, `sp500`, `msft`

**`<model>`** — one of: `original`, `global_feature_atn`, `local_feature_atn`, `time_frame_atn`

Example — reproduce all results on DAX:

```bash
python -W ignore main.py dax original
python -W ignore main.py dax global_feature_atn
python -W ignore main.py dax local_feature_atn
python -W ignore main.py dax time_frame_atn
```

Results and evaluation plots are written to `./Output/results/<market>/`.

---

## Output structure

```
Output/
  ensemble/<market>/<model>/   # per-walk ensemble decisions
  results/<market>/            # evaluation plots and summary CSVs
  q_values/<market>/<model>/   # Q-value logs per epoch and walk
  attentions/<market>/<model>/ # TFA attention weights (time_frame_atn only)
  models/<market>/             # saved DQN weights
```
