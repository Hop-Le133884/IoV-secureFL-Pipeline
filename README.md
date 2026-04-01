# IoV-secureFL-Pipeline
## Secure Federated Learning Pipeline for Unique-Signature Intrusion Detection in IoV Networks

## Overview
A production-ready machine learning pipeline featuring a novel Federated Double Random Forest (Double RF) architecture. This project implements a comprehensive security analysis pipeline for Internet of Vehicles (IoV) networks. By leveraging a two-stage federated ensemble method, vehicles can collaboratively train highly accurate Intrusion Detection Systems (IDS) without ever centralizing sensitive Controller Area Network (CAN) bus logs. 

- **PHASE 1:** Centralized Baseline & Double RF Prototyping
- **PHASE 2:** Privacy-Preserving Federated Double Random Forest architecture by NVIDIA FLARE and XGBoost 
- **PHASE 3:** AWS EMR + S3

# Project Tree

```bash
(iov-securefl-pipeline) hople@hople-zenux:~/working_folder/IoV-secureFL-Pipeline$ tree
.
├── =2.1.0
├── data
│   ├── CICIoV2024.csv
│   ├── processed
│   │   ├── df_federated_5x.csv
│   │   └── IoV_rawAllClasses.csv
│   └── raw
│       ├── data_description.csv
│       ├── dataset_info.md
│       ├── decimal_benign.csv
│       ├── decimal_DoS.csv
│       ├── decimal_spoofing-GAS.csv
│       ├── decimal_spoofing-RPM.csv
│       ├── decimal_spoofing-SPEED.csv
│       └── decimal_spoofing-STEERING_WHEEL.csv
├── data_split_gen.sh
├── jobs
│   └── random_forest_base
│       ├── app
│       │   ├── config
│       │   │   ├── config_fed_client.json
│       │   │   └── config_fed_server.json
│       │   └── custom
│       │       ├── iov_data_loader.py
│       │       └── iov_executor.py
│       └── meta.json
├── jobs_gen.sh
├── LICENSE
├── main.py
├── notebooks
│   └── 01_reproducing_exploration_baseline.ipynb
├── pyproject.toml
├── README.md
├── run_experiment_simulator.sh
├── src
│   └── federated
│       └── meta_model.py
├── testCUDA_imports.py
├── utils
│   ├── model_validation.py
│   ├── prepare_data_split.py
│   └── prepare_job_config.py
└── uv.lock

13 directories, 31 files
```

---

# 🏗️ Phase 1: Centralized Baseline & Double RF Prototyping

In this initial phase, we establish a performance baseline using a centralized machine learning approach. The goal is to validate the Double Random Forest (Double RF) logic—a two-stage ensemble method—before transitioning to a distributed federated environment.

## Description

Phase 1 focuses on data exploration, preprocessing, and the implementation of a sequential **Expert-Master** model architecture. By training a binary classifier (Stage 1) to identify anomalies and passing its probability scores to a multi-class classifier (Stage 2), we enhance the model's ability to distinguish between complex, physically correlated attack vectors like RPM and Speed spoofing.

## Key Components

- **Dataset:** Local instance of `CICIoV2024.csv`
- **Architecture:** Sequential Double Random Forest (XGBoost)
- **Environment:** Research-oriented Jupyter Notebooks
- **Objective:** Maximize detection accuracy and establish baseline metrics (Accuracy, F1-Score, LogLoss) for comparison with later phases.

## Instructions

Before starting, make sure you have the following installed on your system:

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.12+ | `python3 --version` |
| uv | latest | `uv --version` |

### Step 2 — Install uv (Python Package Manager)

This project uses **uv** for fast, reproducible dependency management.

**Linux / macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

After installation, reload your shell:
```bash
source ~/.bashrc   # or source ~/.zshrc
```

Verify:
```bash
uv --version
```

---

### Step 3 — Create the Root Virtual Environment

From the project root, create and activate a virtual environment with Python 3.12:

```bash
uv venv --python 3.12
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install root dependencies (Flower, matplotlib, numpy, pandas):

```bash
uv sync
```

### 1. Patching nvflare xgboost

Ensure you run this after steps above. Because NVflare XGBoost imports all three
recipe classes including XGBHorizontalRecipe (histogram). That module imports
TBAnalyticsReceiver -> torch.utils.tensorboard -> tensorboard, even though
this project only uses tree-based bagging and never touches TensorBoard.

torch and tensorboard is heavy python packages

```bash
python utils/patch_nvflare.py
```

### 2. Dataset Placement

Ensure the raw dataset is available in the expected directory:

```text
./data/CICIoV2024.csv
```

### 3. Launch the Research Notebook

Navigate to the notebooks directory and open the baseline experiment:

```bash
jupyter notebook notebooks/01_reproducing_exploration_baseline.ipynb
```

### 4. Execute the Experiment

Within the notebook, follow the documented cells to:

- **Load & Clean:** Preprocess CAN bus sensor data.
- **Train Stage 1:** Build the Binary Expert to generate the `prob_ATTACK` feature.
- **Train Stage 2:** Build the 6-Class Master model using the augmented feature set.
- **Evaluate:** Generate the baseline confusion matrix and classification report.

---

# 🔄 Phase 2: Privacy-Preserving Federated Double Random Forest

In this phase, the centralized logic from Phase 1 is transformed into a decentralized, privacy-preserving pipeline. Using NVIDIA FLARE as the orchestration engine, we distribute the training process across five simulated vehicle sites.

## Description

Phase 2 implements the Federated Double Random Forest (Double RF) architecture. The core innovation is the ability to achieve high-accuracy intrusion detection while ensuring that raw CAN bus data never leaves the vehicle's local environment. Only encrypted model weights (XGBoost trees) are communicated to the central server.

## Key Components

- **Orchestration:** NVIDIA FLARE (NVFlare) Simulator
- **Engine:** Federated XGBoost with Bagging aggregation
- **Communication Efficiency:** Full convergence in exactly two communication rounds (one for Binary Expert and one for Multi-Class Master)
- **Privacy:** Local data remains isolated; the server only sees aggregated tree structures

## Instructions

### 1. Data Partitioning (Stratified Splitting)

Before starting the federated process, the dataset must be split into isolated site-specific partitions. This script ensures each vehicle receives a balanced distribution of all attack classes:

```bash
bash data_split_gen.sh ./data
```

### 2. Generate Federated Job Configurations

Define global parameters for the federated run, such as number of trees per site, maximum depth, and data split paths:

```bash
bash jobs_gen.sh ./data
```

### 3. Execute the Federated Simulator

Launch the NVFlare Simulator to orchestrate the two-stage training. This process creates a workspace directory, spin up five virtual clients, and execute Stage 1 and Stage 2 sequentially:

```bash
# Clean previous artifacts and launch
rm -rf workspace_iov_double_rf
bash run_experiment_simulator.sh
```

### 4. Global Model Validation

After the simulator finishes, final aggregated models (`xgboost_model_inner.json` and `xgboost_model_outer.json`) are stored on the server. Run validation utility against the unseen holdout set:

```bash
python utils/model_validation.py
```
# ☁️ Phase 3: Cloud Orchestration (AWS EMR + S3)
COMING SOON....
Note: This phase is currently in the architectural planning stage and represents the next milestone for the IoV-secureFL-Pipeline.