# IoV-secureFL-Pipeline: Stage 1

**Secure Federated Learning Pipeline for Unique-Signature Intrusion Detection in IoV Networks**

A cleaned, production-ready implementation of Stage 1, featuring successfully validated double Random Forest experiments on the CICIoV2024 datasets. This stage serves as the foundational baseline and prerequisite for Stage 2, which extends the architecture with distributed federated learning using NVIDIA FLARE and XGBoost.

---

## 📋 Overview

This project implements a comprehensive security analysis pipeline for Internet of Vehicles (IoV) networks using federated learning techniques. **Stage 1** focuses on establishing a robust baseline through centralized machine learning experiments using ensemble methods (specifically Double Random Forest) on real-world vehicular network traffic data.

### Key Accomplishments (Stage 1)

✅ **Clean, validated codebase** - Production-ready implementation with organized structure  
✅ **Double Random Forest experiments** - Successfully executed and benchmarked on CICIoV2024 datasets  
✅ **Multi-site vehicle data processing** - Integrated data from 5 vehicle sites with proper train/test splits  
✅ **Preprocessed datasets** - Cleaned and normalized benign and attack traffic data  
✅ **Baseline establishment** - Metrics and results for comparison with Stage 2 federated approach  

---

## 🎯 Stage 1: Baseline & Prerequisites

Stage 1 establishes the foundational model architecture and dataset baseline:

- **Model Architecture**: Double Random Forest ensemble (centralized training)
- **Dataset**: CICIoV2024 (Intrusion Detection for IoV networks)
- **Attack Classes**: DoS, Spoofing (GAS, RPM, SPEED, STEERING_WHEEL), Benign traffic
- **Training Data**: 5 vehicle site datasets for comprehensive evaluation
- **Metrics**: Classification performance benchmarks (accuracy, precision, recall, F1-score)

This stage validates that the machine learning approach is sound before distributing the training process across federated nodes in Stage 2.

---

## 📁 Project Structure

```
IoV-secureFL-Pipeline/
├── data/
│   ├── raw/                          # Original dataset files
│   │   ├── decimal_benign.csv
│   │   ├── decimal_DoS.csv
│   │   ├── decimal_spoofing-*.csv   # Various spoofing attack types
│   │   ├── data_description.csv
│   │   └── dataset_info.md
│   └── processed/                    # Cleaned & normalized datasets
│       ├── df_federated_5x.csv       # Combined federated format
│       ├── IoV_rawAllClasses.csv
│       └── vehicle_site-*.csv        # Individual site training data (5 sites)
├── notebooks/
│   └── 01_reproducing_exploration_baseline.ipynb  # Experiment notebooks
├── src/
│   └── federated/
│       └── meta_model.py             # Model implementations
├── main.py
├── pyproject.toml                    # Project dependencies
└── README.md
```

---

## 📊 Datasets

**CICIoV2024 - Intrusion Detection for IoV Networks**

### Raw Data
- `decimal_benign.csv` - Benign vehicular network traffic
- `decimal_DoS.csv` - Denial of Service attacks
- `decimal_spoofing-GAS.csv` - Gas pedal spoofing attacks
- `decimal_spoofing-RPM.csv` - RPM gauge spoofing
- `decimal_spoofing-SPEED.csv` - Speed/odometer spoofing
- `decimal_spoofing-STEERING_WHEEL.csv` - Steering wheel spoofing

### Processed Data
- **Multi-site training sets**: `vehicle_site-1_train.csv` through `vehicle_site-5_train.csv`
- **Federated format**: `df_federated_5x.csv` (aligned for distributed training)
- **Consolidated**: `IoV_rawAllClasses.csv` (all samples with labels)

---

## 🚀 Getting Started

### Requirements

- Python 3.12+
- pip or UV package manager

### Installation

1. **Clone the repository**
   ```bash
   cd IoV-secureFL-Pipeline
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```
   
   Or with UV:
   ```bash
   uv pip install -e .
   ```

### Dependencies

Core libraries:
- **Machine Learning**: scikit-learn, lightgbm, xgboost
- **Data Processing**: pandas, numpy, scipy, pyarrow
- **Visualization**: matplotlib, seaborn, shap
- **Federated Learning**: nvflare
- **Utilities**: omegaconf, tqdm, cryptography, kafka-python-ng, jupyter

---

## 📈 Experiments

### Stage 1: Double Random Forest Baseline

Run the Stage 1 baseline experiments:

```bash
jupyter notebook notebooks/01_reproducing_exploration_baseline.ipynb
```

This notebook contains:
- Data loading and preprocessing
- Double Random Forest model training
- Cross-validation experiments
- Performance metric evaluation
- Results visualization and analysis

---

## 🔄 Stage 2: Federated Learning (Coming Next)

Stage 2 will extend this baseline to distributed federated learning:

- **Architecture**: NVIDIA FLARE-based federated learning
- **Model**: XGBoost with federated aggregation
- **Training**: Distributed across multiple virtual sites
- **Goal**: Compare centralized (Stage 1) vs federated (Stage 2) performance

The results from Stage 1 serve as the critical performance baseline for Stage 2 validation.

---

## 📝 License

See [LICENSE](LICENSE) file for details.

---

## 🔗 Related Work

This project is based on research in secure federated learning for vehicular networks and intrusion detection systems. It extends prior work with a modern, production-ready implementation pipeline.

---

## 📞 Notes

- **CUDA Support**: The pipeline is optimized for GPU acceleration. Use `testCUDA_imports.py` to verify CUDA availability.
- **Configuration**: Project settings managed via `omegaconf` for easy experimentation.
- **Kafka Integration**: Supports real-time vehicular data streaming via Kafka (infrastructure optional).

