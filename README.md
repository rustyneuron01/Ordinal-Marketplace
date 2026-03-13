<h1 align="center">
    BTC & ETH & SOL Price Predict
</h1>

---

## Summary

This project produces **ensemble forecasts** of future asset prices (many simulated paths per request) to capture full **probability distributions** of price movement, not just point estimates. Output is synthetic price data for training AI agents and for options pricing and portfolio risk analytics. Quality is measured with the **Continuous Ranked Probability Score (CRPS)** over ensemble predictions against realized prices, with focus on **calibration and sharpness** (e.g. volatility clustering, fat tails).

---

### Table of contents

- [1. Overview](#-1-overview)
  - [1.1. Introduction](#11-introduction)
  - [1.2. Task Presented to Workers](#12-task-presented-to-workers)
  - [1.3. Coordinator's Scoring Methodology](#13-coordinators-scoring-methodology)
  - [1.4. Calculation of Leaderboard Score](#14-calculation-of-leaderboard-score)
  - [1.5. Overall Purpose](#15-overall-purpose)
- [2. Tech stack](#2-tech-stack)
- [3. Usage](#-3-usage)
- [4. License](#-4-license)

---

## 🔭 1. Overview

### 1.1. Introduction

This system provides high-quality synthetic price data and probabilistic forecasting. **Workers** generate multiple simulated price paths per request; paths must reflect real-world dynamics (volatility clustering, fat-tailed distributions). **Coordinators** score workers using the Continuous Ranked Probability Score (CRPS), which measures both calibration and sharpness of forecasts against actual price movements. Recent performance is weighted more heavily; emissions are allocated by relative performance.

The system aims to be a key source of synthetic price data for AI agents and for options trading and portfolio management.

### 1.2. Task Presented to Workers

Workers provide **probabilistic forecasts** of future price movements: multiple simulated price paths per asset over specified time increments and horizon. Current request format: **1000 simulated paths** for BTC, ETH, SOL, XAU (and tokenized equities SPYX, NVDAX, TSLAX, AAPLX, GOOGLX) over 24 hours at 5-minute increments. The system focuses on **quantifying uncertainty**—paths should represent the worker’s view of the probability distribution and encapsulate realistic dynamics (volatility clustering, fat tails).

**Prompt parameters:** (start_time, asset, time_increment, time_horizon, num_simulations)

- **Start Time ($t_0$)**: 1 minute from the time of the request.
- **Asset**: BTC, ETH, XAU, SOL, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX (CRPS per asset contributes to final worker weights).
- **Time Increment ($\Delta t$)**: 5 minutes.
- **Time Horizon ($T$)**: 24 hours.
- **Number of Simulations ($N_{\text{sim}}$)**: 1000.

**Asset Weights**  
BTC 1.0 · ETH 0.67 · XAU 2.26 · SOL 0.59 · SPYX 2.99 · NVDAX 1.39 · TSLAX 1.42 · AAPLX 1.86 · GOOGLX 1.43

Coordinators send requests (e.g. BTC/ETH at 30 min intervals). The worker has until the start time to return $N_{\text{sim}}$ paths. Use the Pyth Oracle (or equivalent) for the asset price at start_time. Late or invalid responses are scored 0 for that prompt.

**1-Hour HFT:** The system also runs a short-horizon task (1 hour, BTC/ETH/SOL/XAU). Emissions split: 50% 24-hour predictions, 50% 1-hour HFT.

### 1.3. Coordinator's Scoring Methodology

After the time horizon has passed, coordinators compare each worker’s predicted paths to realized prices using the **Continuous Ranked Probability Score (CRPS)**. CRPS is a proper scoring rule for continuous variables (calibration + sharpness). Lower CRPS = better forecast.

#### Application of CRPS to Ensemble Forecasts

Workers produce **ensemble forecasts** (a finite number of simulated paths). For observation $x$ and ensemble $y_1, \dots, y_N$:

$$
\text{CRPS} = \frac{1}{N}\sum_{n=1}^N \left| y_n - x \right| - \frac{1}{2N^2} \sum_{n=1}^N \sum_{m=1}^N \left| y_n - y_m \right|
$$

The first term is average absolute error vs. observation; the second term accounts for ensemble spread. CRPS is computed on **price change in basis points** per interval so scores are comparable across assets.

#### Application to Multiple Time Increments

CRPS is applied over several intervals (e.g. 5 min, 30 min, 3 h, 24 h). For each increment: predicted price changes (from worker paths), observed price changes (from real prices, e.g. Pyth at each step), then CRPS. The **final score for a worker for one prompt** is the sum of CRPS over all increments.

### 1.4. Calculation of Leaderboard Score

#### CRPS Transformation

- Rank workers by CRPS sum; cap worst 10% at 90th percentile.
- Subtract the best (lowest) CRPS from all scores so the best worker gets 0.
- Workers that failed to submit valid predictions in time get the 90th percentile score.

#### Rolling Average (Leaderboard Score)

Coordinators store per-request scores and compute a **rolling average** over the past 10 days, weighted by asset. Leaderboard score for worker $i$ at time $t$:

$$
L_i(t) = \frac{\sum_{j} S_{i,j} w_{k,j}}{\sum_{j} w_{k,j}}
$$

($S_{i,j}$ = score of worker $i$ at request $j$; $w_{k,j}$ = asset weight. Sum over $j$ with $t - t_j \leq T$, $T = 10$ days.) Lowest score = highest rank.

#### Final Emissions

$$
A_i(t) = \frac{e^{-\beta \cdot L_i(t)}}{\sum_j e^{-\beta \cdot L_j(t)}} \cdot E(t)
$$

with $\beta = -0.1$ and $E(t)$ total emission at time $t$.

### 1.5. Overall Purpose

1. **CRPS scoring** — Objective measure of forecast quality across time increments.
2. **Ensemble forecasts** — CRPS from finite simulations.
3. **Multiple time increments** — Short- and long-term evaluation.
4. **Moving average** — Rewards consistency.
5. **Softmax allocation** — Emissions proportional to performance.

---

## 2. Tech stack

| Category               | Technologies                                           |
| ---------------------- | ------------------------------------------------------ |
| **Language & runtime** | Python 3.11                                            |
| **ML / volatility**    | XGBoost (volatility prediction from cached features)   |
| **Numerics & scoring** | NumPy, Pandas, properscoring (CRPS)                    |
| **Price data**         | Pyth (Hermes) API; cached data for volatility training |
| **Backend & API**      | Pydantic, async request handling                       |
| **Data & storage**     | PostgreSQL, SQLAlchemy, Alembic                        |
| **Observability**      | Google Cloud Logging, Weights & Biases (optional)      |
| **DevOps**             | Docker, Docker Compose                                 |
| **Testing**            | pytest, CRPS/validation/simulation fixtures            |
