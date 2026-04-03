# IoV-secureFL-Pipeline
## Secure Federated Learning Pipeline for Unique-Signature Intrusion Detection in IoV Networks

## Overview

A production-ready machine learning pipeline featuring a novel **Federated Double Random Forest (Double RF)** architecture. This project implements a comprehensive security analysis pipeline for Internet of Vehicles (IoV) networks. By leveraging a two-stage federated ensemble method, vehicles can collaboratively train highly accurate Intrusion Detection Systems (IDS) without ever centralizing sensitive Controller Area Network (CAN) bus logs.

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Centralized Baseline & Double RF Prototyping | Complete |
| **Phase 2** | Privacy-Preserving Federated Double Random Forest (NVFlare + XGBoost) | Complete |
| **Phase 3** | Real Distributed Deployment (AWS EMR + S3, Non-IID) | Complete |

---

# Project Structure

```
IoV-secureFL-Pipeline/
├── data/
│   ├── raw/                         # Original CICIoV2024 CSVs per class
│   └── processed/                   # Deduplicated + federated splits
├── jobs/
│   └── random_forest_base/          # NVFlare job template (executor + config)
├── aws_emr/                         # Phase 3 — AWS EMR deployment workspace
│   ├── config/emr_config.json
│   ├── data/prepare_noniid_split.py
│   ├── fl/                          # S3-aware executor + data loader
│   ├── emr/                         # Cluster JSON templates
│   ├── bootstrap/                   # EMR bootstrap script
│   └── scripts/                     # 01–05 step scripts
├── notebooks/
│   └── 01_reproducing_exploration_baseline.ipynb
├── utils/
│   ├── model_validation.py
│   ├── generate_dp_report.py        # DP sweep runner (multi-seed, saves CSV)
│   └── prepare_job_config.py
├── jobs_gen.sh                      # Generate NVFlare job configs
├── run_experiment_simulator.sh      # Launch NVFlare simulator
└── cleansing_job.sh                 # Remove generated artifacts
```

---

# Phase 1 — Centralized Baseline & Double RF Prototyping

Establishes a performance baseline using centralized machine learning to validate the Double RF architecture before transitioning to a federated environment.

## Architecture

Two-stage sequential **Expert-Master** model:
- **Stage 1 (Binary Expert):** Binary classifier — detects attack vs benign
- **Stage 2 (Multi-class Master):** 6-class classifier augmented with `prob_ATTACK` from Stage 1

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

**Patch NVFlare** (removes unused torch/tensorboard imports that slow down installation):
```bash
python utils/patch_nvflare.py
```

## Dataset

Place the raw dataset at:
```
./data/CICIoV2024.csv
```

## Run

```bash
jupyter notebook notebooks/01_reproducing_exploration_baseline.ipynb
```

Follow the notebook cells:
1. Load & Clean — preprocess CAN bus sensor data
2. Train Stage 1 — binary expert → generates `prob_ATTACK` feature
3. Train Stage 2 — 6-class master on augmented feature set
4. Evaluate — confusion matrix, classification report (baseline F1, Accuracy, LogLoss)

---

# Phase 2 — Privacy-Preserving Federated Double Random Forest

Distributes the Double RF across 5 simulated vehicle sites using **NVIDIA FLARE**. Raw CAN bus data never leaves each vehicle — only XGBoost tree structures are communicated to the server.

## Architecture

- **Orchestration:** NVFlare Simulator (5 virtual client sites)
- **Aggregation:** XGBoost Bagging (`XGBBaggingAggregator`) — concatenates trees from all sites
- **Rounds:** 2 communication rounds (Stage 1 binary → Stage 2 6-class)
- **Privacy:** Local data stays isolated; server only sees aggregated tree weights

## Step 1 — Data Partitioning

Splits the deduplicated dataset into 5 stratified site partitions (IID):

```bash
bash data_split_gen.sh ./data
```

Output: `data/IoV/data_splits/data_site-{1..5}.json`

---

## Experiment A — Without Differential Privacy

Trains the Federated Double RF with no noise added to leaf values.

### Step 2 — Generate Job Config

```bash
bash jobs_gen.sh ./data
```

### Step 3 — Run Simulator

```bash
bash run_experiment_simulator.sh
```

### Step 4 — Validate Global Model

```bash
python utils/model_validation.py
```

Output: macro F1, accuracy, per-class breakdown against the holdout test set.

---

## Experiment B — With Differential Privacy (ε, δ)-DP

Adds calibrated Gaussian noise to XGBoost leaf values after local training.
Mechanism: **σ = C · √(2 ln(1.25/δ)) / ε** where C = clip bound (L∞ sensitivity).

### Option B.1 — Single Run (one ε, one seed)

```bash
# Step 2: Generate job config with ε=80, seed=42
DP_EPSILON=80 SEED=42 bash jobs_gen.sh ./data

# Step 3: Run simulator
bash run_experiment_simulator.sh

# Step 4: Validate
python utils/model_validation.py
```

### Option B.2 — Full DP Sweep (multi-ε, multi-seed, saves CSV report)

Runs each ε value 3 times with seeds 42, 456, 1000. Computes mean ± std for F1 and Accuracy. Saves full tradeoff table to `reports/dp_tradeoff.csv`.

**Expected runtime: ~40 minutes**

```bash
python utils/generate_dp_report.py --epsilon "inf; 100; 80; 70; 50; 1"
```

Output CSV columns:
```
epsilon, sigma, f1_seed_42, f1_seed_456, f1_seed_1000, f1_mean, f1_std,
acc_seed_42, acc_seed_456, acc_seed_1000, acc_mean, acc_std
```

Expected DP tradeoff:

| ε | σ | Effect |
|---|---|--------|
| ∞ | 0.000 | No DP — baseline |
| 100 | 0.242 | Tiny noise, near-baseline |
| 80 | 0.303 | Low noise, near-baseline |
| 70 | 0.346 | Low-mid noise |
| 50 | 0.485 | Moderate noise, noticeable drop |
| 1 | 24.22 | Extreme noise — model collapses |

### Option B.3 — Visualize DP Tradeoff

```bash
python utils/dpReport_visualization.py
```

Generates plots: F1 & Accuracy vs ε (log scale, inverted x-axis) and F1 vs σ, saved to `reports/`.

---

## Cleanup

Remove all generated artifacts (workspace, job configs, reports):

```bash
bash cleansing_job.sh           # dry run — shows what would be deleted
bash cleansing_job.sh --confirm # actually delete
```

---

# Phase 3 — AWS EMR + S3 (Real Distributed Deployment)
# DETAIL in /IoV-secureFL-Pipeline_awsEC2S3/README.md

Deploys the same Double RF pipeline on real distributed AWS infrastructure with **Non-IID data splitting** (Dirichlet distribution) to simulate realistic vehicle fleet heterogeneity.

## Key Differences from Phase 2

| | Phase 2 | Phase 3 |
|---|---|---|
| Infrastructure | NVFlare simulator, single machine | 6 EMR clusters (1 server + 5 clients) |
| Data split | IID stratified | Non-IID Dirichlet (BENIGN α=2.0, attack α=0.5) |
| Data location | Local CSV | S3 bucket |
| DP epsilon | Sweep (1 → ∞) | Fixed ε=80 |
| Dataset | ~5,493 rows (same processed data) | Same deduplicated dataset |

## Non-IID Split Design

Uses **class-specific Dirichlet concentration**:
- **BENIGN (α=2.0)** — majority class (96% of data), needs higher α to avoid extreme site imbalance
- **Attack classes (α=0.5)** — heterogeneous by design, simulates vehicles in different environments