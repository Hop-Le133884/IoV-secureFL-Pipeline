# IoV Secure Federated Learning Pipeline — AWS EC2

A privacy-preserving Federated Learning pipeline for Internet-of-Vehicles (IoV) intrusion detection.
Uses **NVIDIA FLARE (NVFlare)** with a **Double Random Forest** (XGBoost, two-stage) trained across 5 EC2 vehicle nodes, protected by **Differential Privacy** (Gaussian mechanism, ε=80, δ=1e-5).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│              AWS VPC (Private Subnet)                │
│                                                      │
│  ┌───────────────────┐      ┌──────────────────────┐ │
│  │  Master Server    │      │   5 × Vehicle Nodes  │ │
│  │  (FL Server +     │◄────►│   (FL Clients)       │ │
│  │   Admin Console)  │      │   site-1 … site-5    │ │
│  │  172.31.33.187    │      │   172.31.71.9        │ │
│  │                   │      │   172.31.67.199      │ │
│  │  Port 8002 (FL)   │      │   172.31.76.174      │ │
│  │  Port 8003 (Admin)│      │   172.31.77.237      │ │
│  └───────────────────┘      │   172.31.64.105      │ │
│                             └──────────────────────┘ │
└──────────────────────────────────────────────────────┘

Double Random Forest (two-stage pipeline):

  ┌─────────────────────────────────────────────────────────────┐
  │  Stage 1 — train_inner  (Binary RF)                        │
  │  Input : raw CAN-bus features                              │
  │  Output: prob_ATTACK  (probability of being an attack)     │
  └──────────────────────────┬──────────────────────────────────┘
                             │  prob_ATTACK appended as feature
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Stage 2 — train_outer  (6-Class RF)                       │
  │  Input : raw features  +  prob_ATTACK                      │
  │  Output: BENIGN / DOS / RPM / SPEED / STEERING_WHEEL / GAS │
  └─────────────────────────────────────────────────────────────┘
```

---

## Part 1 — AWS Infrastructure Setup (Web Console)

### 1.1 Create a Key Pair

1. EC2 → **Key Pairs** → **Create key pair**
2. Name: `iov-dp-key`
3. Type: RSA, format: `.pem`
4. Download and store at `ec2Key/iov-dp-key.pem` inside the repo
5. Set permissions: `chmod 400 ec2Key/iov-dp-key.pem`

### 1.2 Create a VPC and Subnet

1. VPC → **Create VPC**
   - Name: `iov-securefl-vpc`
   - IPv4 CIDR: `172.31.0.0/16`
2. Create a **Subnet** inside the VPC
   - Name: `iov-securefl-subnet`
   - CIDR: `172.31.0.0/24`
   - Enable **Auto-assign public IPv4**

> If using the default VPC (172.31.0.0/16), you can skip creating a new VPC.

### 1.2.1 Attach an Internet Gateway (required for SSH access)

Without this step, your instances have public IPs but are **unreachable from the internet** — SSH and nc will time out even if the security group is correct.

1. VPC → **Internet Gateways** → **Create internet gateway**
   - Name: `iov-securefl-igw`
   - Create

2. **Actions → Attach to VPC** → select `iov-securefl-vpc` → Attach

   > ⚠️ Do not reuse an IGW from another VPC — you will get a "belong to different networks" error.

3. VPC → **Route Tables** → find the route table associated with `iov-securefl-subnet` → **Routes** tab → **Edit routes** → **Add route**:

   | Destination | Target |
   |---|---|
   | `0.0.0.0/0` | `iov-securefl-igw` (igw-xxxxxxxx) |

   → Save changes

4. Verify: the route table should now show two routes:

   | Destination | Target |
   |---|---|
   | `172.31.0.0/16` | local |
   | `0.0.0.0/0` | igw-xxxxxxxx |

### 1.3 Create a Security Group

1. EC2 → **Security Groups** → **Create security group**
2. Name: `iov-securefl-sg`
3. VPC: select your VPC above
4. **Inbound Rules** — add all of:

| Type        | Protocol | Port  | Source           | Purpose              |
|-------------|----------|-------|------------------|----------------------|
| SSH         | TCP      | 22    | My IP / 0.0.0.0/0 | SSH access        |
| Custom TCP  | TCP      | 8002  | 172.31.0.0/16    | NVFlare FL port      |
| Custom TCP  | TCP      | 8003  | 172.31.0.0/16    | NVFlare Admin port   |
| All traffic | All      | All   | 172.31.0.0/16    | Intra-VPC (clients ↔ server) |

5. **Outbound Rules**: Allow all (default)

### 1.4 Launch EC2 Instances

Launch **6 instances** total: 1 master server + 5 vehicle clients.

**For each instance:**

1. EC2 → **Launch Instance**
2. **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
3. **Instance type**:
   - Master server: `5.xlarge` (or testing on lower config)
   - Vehicle clients: `m4.large` (or testing on lower config)
4. **Key pair**: `iov-dp-key`
5. **Network**: select your VPC and subnet
6. **Security group**: `iov-securefl-sg`
7. **Storage**: 25 GB gp3 (master), 20 GB gp3 (clients) (or testing on lower config)
8. **Advanced → User data** leave blank; provisioning is done via scripts

After launch, note the **private IPv4 addresses** for all 6 instances.

### 1.5 Update Instance IPs in Scripts

Edit the following files and replace the IPs with your actual private IPv4 addresses:

```
fleet_deployment.sh   → CORE_IPS
network_provision.sh  → CORE_IPS
start_fleet.sh        → CORE_IPS
clean_fleet.sh        → CORE_IPS
monitor_fleet.sh      → CORE_IPS
jobs_gen.sh           → CORE_IPS
```

Current configured IPs (update these):
- Master server: `172.31.33.187`
- site-1: `172.31.71.9`
- site-2: `172.31.67.199`
- site-3: `172.31.76.174`
- site-4: `172.31.77.237`
- site-5: `172.31.64.105`

### 1.6 Verify Connectivity

**Step 1 — SSH from your local machine to the master node:**

```bash
# From your local machine
ssh -i ec2Key/iov-dp-key.pem ubuntu@<MASTER_PUBLIC_IP>
```

> Get the master's public IP from EC2 Console → Instances → select master instance → **Public IPv4 address**.
> Note: public IP changes on every stop/start unless you attach an Elastic IP.

**Step 2 — Copy the project folder from your local machine to the master node:**

Run this from your **local machine** (not inside the SSH session):

```bash
# Copy project code
rsync -avz -e "ssh -i ec2Key/iov-dp-key.pem" \
    ../IoV-secureFL-Pipeline_awsEC2/ \
    ubuntu@<MASTER_PUBLIC_IP>:~/IoV-secureFL-Pipeline_awsEC2/

# # Copy processed data (required for jobs_gen.sh to generate data splits)
# rsync -avz -e "ssh -i ec2Key/iov-dp-key.pem" \
#     ./data/processed/ \
#     ubuntu@<MASTER_PUBLIC_IP>:~/IoV-secureFL-Pipeline_awsEC2/data/processed/
# ```

> Re-run this any time you update scripts locally — rsync only transfers changed files.

**Step 3 — From the master node, test SSH to each client:**

```bash
ssh -i ec2Key/iov-dp-key.pem ubuntu@172.31.71.9 "echo site-1 OK"
ssh -i ec2Key/iov-dp-key.pem ubuntu@172.31.67.199 "echo site-2 OK"
# ... etc
```

---

## Part 2 — Project Setup (Master Node)

All commands run on the **master server** from inside `~/IoV-secureFL-Pipeline_awsEC2/`.

### 2.1 Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2.2 Clone / Transfer the Repository

The project should already be on the master node from Step 2 in section 1.6.
If you need to re-transfer or update:

Then on the master node:

```bash
cd ~/IoV-secureFL-Pipeline_awsEC2
```

### 2.3 Prepare the Dataset

The pipeline expects a pre-processed federated dataset at:
```
data/processed/df_federated_100x.csv
```

Run the notebook `01_reproducing_exploration_baseline.ipynb` to generate it, or transfer it from your local machine.

---

## Part 3 — Full Pipeline Execution

Run these steps in order from the **master node**, inside `~/IoV-secureFL-Pipeline_awsEC2/`.

### Step 1 — Clean Master Node (fresh start)

```bash
./cleansing_job.sh --confirm
```

Removes: `.venv`, generated job configs, NVFlare workspace, model outputs, `/tmp/nvflare/` cache.

### Step 2 — Clean All 5 Client Nodes

```bash
./clean_fleet.sh
```

Wipes `~/IoV-secureFL-Pipeline_awsEC2/` from all 5 EC2 client nodes and kills any running NVFlare processes.

### Step 3 — Install Python Dependencies (Master)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync # Install all packages and dependencies
```

### Step 4 — Generate Data Splits

```bash
bash data_split_gen.sh ./data
```

Splits `data/processed/df_federated_100x.csv` into:
- `data/IoV/data_splits/data_site-{1..5}.json` — per-site data pointers
- `data/processed/df_server_test.csv` — held-out server test set

Uses class-specific Dirichlet allocation to simulate non-IID distribution across sites.
See **Non-IID Data Distribution Strategy** section below for full details.

### Step 5 — Deploy to All 5 Client Nodes

```bash
./fleet_deployment.sh
or 
bash fleet_deployment.sh
```

This:
- Rsyncs the project to all 5 client EC2 nodes
- Installs `uv`, creates `.venv`, and installs NVFlare + XGBoost + dependencies on each client

### Step 6 — Provision NVFlare Network

```bash
bash network_provision.sh
```

This:
- Generates `project.yml` with the current master IP
- Runs `nvflare provision` to create PKI certificates and startup kits
- Patches server `start.sh` with `/etc/hosts` fix for hostname resolution
- Distributes `site-N/` startup kits to each client node via SCP

> **Must run before `jobs_gen.sh`.**

### Step 7 — Generate and Deploy Job Configuration

```bash
DP_EPSILON=80 SEED=42 bash jobs_gen.sh ./data
```

This:
- Generates job config under `jobs/iov_double_rf_5_sites/`
- Syncs the job to the NVFlare admin transfer directory
- Writes the correct executor to all 5 site app directories
- Pushes per-site data split JSON files to each client node

**DP parameters:**

| Parameter     | Value   | Description                      |
|---------------|---------|----------------------------------|
| `DP_EPSILON`  | `80`    | Privacy budget ε (higher = less noise) |
| `DP_DELTA`    | `1e-5`  | Failure probability δ            |
| `DP_CLIP_BOUND` | `5.0` | Leaf value clipping bound C      |
| `SEED`        | `42`    | Random seed for XGBoost + DP noise |

To disable DP: `SEED=42 bash jobs_gen.sh ./data` (omit `DP_EPSILON`)

### Step 8 — Start the FL Server

```bash
bash workspace/iov_securefl_network/prod_00/server/startup/start.sh
```

Server starts in background. Logs to:
```
workspace/iov_securefl_network/prod_00/server/log.txt
```

Verify server is running:
```bash
tail -f workspace/iov_securefl_network/prod_00/server/log.txt
```

### Step 9 — Start All 5 Vehicle Clients

```bash
./start_fleet.sh
```

SSHs into each client node and starts the NVFlare client process. Each client logs to `~/IoV-secureFL-Pipeline_awsEC2/client.log`.

### Step 10 — Submit the FL Job (Admin Console)

```bash
bash workspace/iov_securefl_network/prod_00/admin@master.com/startup/fl_admin.sh
```
User Name: admin@master.com


Inside the admin console:

```
> check_status server          # verify all 5 clients are registered
> submit_job jobs/iov_double_rf_5_sites
> check_status server          # monitor job progress
```

The job runs two sequential federated rounds:
1. **Stage 1** (`train_inner`): Binary XGBoost RF — all 5 sites train and aggregate
2. **Stage 2** (`train_outer`): 6-class XGBoost RF — uses cached Stage 1 model

Expected server log output:
```
Saving received model to .../xgboost_model_inner.json   ← Stage 1 complete
Saving received model to .../xgboost_model_outer.json   ← Stage 2 complete
```

---

## Part 4 — Monitoring and Evaluation

### Monitor Client Training

```bash
# Snapshot of all 5 sites
./monitor_fleet.sh

# Live tail site-1
./monitor_fleet.sh live 1

# Live tail site-3
./monitor_fleet.sh live 3
```

Expected client output during training:
```
======> site-1 Stage 1 | Acc: 99.21% | Macro-F1: 72.34% | LogLoss: 0.018234 <======
======> site-1 Stage 2 | Acc: 98.84% | Macro-F1: 65.12% | LogLoss: 0.042100 <======
```

### Evaluate Global Models

```bash
# Latest completed job
./evaluate.sh

# Specific job ID
./evaluate.sh <job_id>
```

Example output:
```
Example output:
```
============================================================
      DOUBLE RANDOM FOREST — EVALUATION ON UNIQUE SIGNATURES
============================================================
  Overall Accuracy:  99.6656%
  Overall LogLoss:   0.029376
  Macro F1-Score:    65.4666%

Classification Report:
                precision    recall  f1-score   support

        BENIGN       1.00      1.00      1.00      3547
           DOS       0.95      1.00      0.98        21
           GAS       1.00      0.50      0.67         2
           RPM       0.56      1.00      0.71        10
         SPEED       1.00      0.40      0.57         5
STEERING_WHEEL       0.00      0.00      0.00         3

      accuracy                           1.00      3588
     macro avg       0.75      0.65      0.65      3588
  weighted avg       1.00      1.00      1.00      3588


Confusion Matrix (Rows: True, Columns: Predicted):
[[3542    1    0    4    0    0]
 [   0   21    0    0    0    0]
 [   0    0    1    1    0    0]
 [   0    0    0   10    0    0]
 [   0    0    0    3    2    0]
 [   3    0    0    0    0    0]]
```

> **Note on Macro F1**: Low Macro F1 (65%) reflects extreme class imbalance in the test set (BENIGN = 98.9%). The model correctly classifies all BENIGN and DOS traffic; rare attack classes (SPEED=5 samples, STEERING_WHEEL=3 samples) have insufficient test support for reliable evaluation.

Models are extracted to `models/`:
- `models/xgboost_model_inner.json` — Stage 1 global binary classifier
- `models/xgboost_model_outer.json` — Stage 2 global 6-class classifier

---

## Part 5 — Clean Re-Run Reference

```bash
# 1. Clean master
./cleansing_job.sh --confirm

# 2. Clean all 5 clients
./clean_fleet.sh

# 3. Install deps
uv sync && source .venv/bin/activate

# 4. Deploy to clients
./fleet_deployment.sh

# 5. Generate data splits
bash data_split_gen.sh ./data

# 6. Provision NVFlare network  ← MUST be before jobs_gen
bash network_provision.sh

# 7. Generate + deploy job config
DP_EPSILON=80 SEED=42 bash jobs_gen.sh ./data

# 8. Start FL server
bash workspace/iov_securefl_network/prod_00/server/startup/start.sh

# 9. Start all clients
./start_fleet.sh

# 10. Admin console → submit job
bash workspace/iov_securefl_network/prod_00/admin@master.com/startup/fl_admin.sh
# Inside console: submit_job iov_double_rf_5_sites

# 11. Evaluate
./evaluate.sh
```

---

## Non-IID Data Distribution Strategy

Real-world IoV deployments are inherently non-IID: vehicles on different routes encounter
different attack surfaces and see different traffic patterns. This pipeline implements
**class-specific Dirichlet partitioning** to faithfully simulate this heterogeneity.

### Dirichlet Allocation

Each class is split across the 5 sites by sampling proportions from a Dirichlet distribution.
The concentration parameter α controls how skewed the split is:

- **α → ∞**: perfectly IID (all sites get equal share)
- **α → 0**: maximally non-IID (one site gets everything, others get nothing)

Two separate α values are used — one for the majority class, one for attack classes:

| Class type    | α value | Effect |
|---------------|---------|--------|
| BENIGN        | `2.0`   | Roughly balanced across sites — prevents pathological splits where a site gets almost no normal traffic |
| Attack classes | `0.5`  | Heterogeneous — sites specialise in different attack subsets; some sites intentionally receive **zero samples** of certain attack classes |

### Why Different α Values?

BENIGN makes up ~96% of all training data. Using a low α for BENIGN would cause some sites
to get as few as 16 rows of normal traffic, making local training numerically unstable.
Using a high α (2.0) keeps BENIGN roughly balanced while still introducing natural variation.

Attack classes are rare and domain-specific (RPM tampering vs. GAS injection vs. DOS).
A low α (0.5) reflects the realistic scenario where a vehicle node only ever sees a subset
of attack types — forcing the federation to learn globally what no single node knows locally.
This is the core motivation for federated learning in this domain.

### Server Test Set

The held-out test set (`df_server_test.csv`) is built by **strict per-class deduplication**:
for each class, only rows with a unique combination of CAN-bus signature columns
(`ID`, `DATA_0`–`DATA_7`) are retained. This ensures evaluation is on genuinely unseen
traffic patterns, not repeated instances from the training distribution.

```
Signature columns: ID, DATA_0, DATA_1, DATA_2, DATA_3, DATA_4, DATA_5, DATA_6, DATA_7
```

### Split Parameters

```bash
bash data_split_gen.sh ./data
# Internally calls:
#   --alpha_benign 2.0   (BENIGN class concentration)
#   --alpha_attack 0.5   (all attack classes concentration)
#   --seed 42            (reproducibility)
```

To inspect the actual per-site class distribution after splitting:

```bash
python3 utils/prepare_data_split.py \
    --federated_data_path ./data/processed/df_federated_100x.csv \
    --site_num 5 \
    --out_path ./data/IoV/data_splits \
    --processed_dir ./data/processed \
    --alpha_benign 2.0 \
    --alpha_attack 0.5 \
    --seed 42
```

The script prints a distribution table like:

```
Non-IID split  (BENIGN α=2.0, attack α=0.5)
Site          BENIGN             DOS             GAS    ...   Total   Benign%   Missing attack classes
------------------------------------------------------------------------------------------------------
site-1         12043               8               0    ...   12102    99.5%    ['GAS', 'SPEED']
site-2         11876              14               2    ...   11934    99.5%    ['STEERING_WHEEL']
...
```

Sites with zero samples of a given attack class must rely entirely on the federated
global model to handle that attack type — demonstrating the necessity of federation.

---

## Model Architecture

### Double Random Forest

| Stage | Task       | Objective          | Trees | Classes |
|-------|------------|--------------------|-------|---------|
| 1     | Inner RF   | `binary:logistic`  | 20    | 2 (Benign/Attack) |
| 2     | Outer RF   | `multi:softprob`   | 20    | 6 (Attack types)  |

Stage 2 uses the Stage 1 predicted attack probability (`prob_ATTACK`) as an additional feature, enabling the outer model to leverage binary classification confidence for multi-class discrimination.

### XGBoost Hyperparameters

| Parameter              | Value |
|------------------------|-------|
| `num_local_parallel_tree` | 20  |
| `max_depth`            | 8     |
| `learning_rate`        | 0.1   |
| `local_subsample`      | 0.8   |
| `nthread`              | 4     |
| `tree_method`          | hist  |

### Differential Privacy

Gaussian mechanism applied to XGBoost leaf values after local training:

```
σ = C · √(2 ln(1.25 / δ)) / ε
```

| Parameter | Value  |
|-----------|--------|
| ε (epsilon) | 80.0 |
| δ (delta)   | 1e-5 |
| C (clip bound) | 5.0 |

Noise is applied per-leaf: each leaf value is clipped to `[-C, C]` then Gaussian noise `N(0, σ²)` is added before the model is sent to the aggregator.

### Federated Aggregation

`XGBBaggingAggregator` — bagging ensemble: all 5 site models are concatenated into a single global XGBoost model (tree bagging, not averaging).

---

## Network Ports

| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 22   | TCP      | Inbound   | SSH (admin access) |
| 8002 | TCP      | Inbound/Outbound | NVFlare FL communication (clients ↔ server) |
| 8003 | TCP      | Inbound   | NVFlare Admin Console |

---

## Project Structure

```
IoV-secureFL-Pipeline_awsEC2/
├── ec2Key/
│   └── iov-dp-key.pem              # AWS EC2 SSH key (chmod 400)
├── data/
│   ├── processed/
│   │   ├── df_federated_100x.csv     # 100x-capped federated training data (input)
│   │   └── df_server_test.csv      # Unique-signature server test set (generated)
│   └── IoV/
│       └── data_splits/
│           └── data_site-{1..5}.json  # Per-site data pointers (generated)
├── jobs/
│   └── iov_double_rf_5_sites/     # Generated FL job config
│       ├── app_server/
│       └── app_site-{1..5}/
│           └── custom/
│               ├── iov_executor.py      # Double RF FL executor
│               └── iov_data_loader.py  # IoV data loader
├── models/                         # Extracted global models (after evaluate.sh)
│   ├── xgboost_model_inner.json
│   └── xgboost_model_outer.json
├── utils/
│   ├── prepare_data_split.py       # Dirichlet data partitioning
│   ├── prepare_job_config.py       # NVFlare job config generator
│   └── model_validation.py         # Evaluation script
├── workspace/                      # NVFlare provisioned workspace (generated)
│   └── iov_securefl_network/
│       └── prod_00/
│           ├── server/             # Server startup kit + logs
│           ├── site-{1..5}/       # Client startup kits (distributed to EC2)
│           └── admin@master.com/  # Admin console startup kit
├── cleansing_job.sh    # Clean master node artifacts
├── clean_fleet.sh      # Wipe all 5 client nodes
├── fleet_deployment.sh # Bootstrap + sync all client nodes
├── data_split_gen.sh   # Generate FL data splits
├── network_provision.sh # NVFlare provisioning + cert distribution
├── jobs_gen.sh         # Generate job config + deploy to transfer dir
├── start_fleet.sh      # Start NVFlare clients on all 5 EC2 nodes
├── monitor_fleet.sh    # Monitor client training output
├── evaluate.sh         # Extract models + run evaluation
└── pyproject.toml      # Python dependencies (managed by uv)
```

---

## Troubleshooting

### Server won't start
```bash
tail -50 workspace/iov_securefl_network/prod_00/server/log.txt
```
- If `start.sh` prints "start fl because process already exists" — server is already running (not an error)
- Logs go to `log.txt`, not to terminal (runs in background)

### Clients not connecting (`ClientConnectorDNSError: server:8003`)
The hostname `server` must resolve on each client. `start_fleet.sh` handles this automatically by adding `<SERVER_IP> server` to `/etc/hosts` before starting each client.

Manual fix on a client:
```bash
echo "172.31.33.187 server" | sudo tee -a /etc/hosts
```

### Job fails at Stage 2 (`EXECUTION_EXCEPTION` on `train_outer`)
This means the wrong `iov_executor.py` was deployed (simulator-mode version with `import os`). Re-run:
```bash
DP_EPSILON=80 SEED=42 bash jobs_gen.sh ./data
```
`jobs_gen.sh` always writes the correct executor to the transfer directory regardless of source file state.

### `submit_job` says folder not found
The job must be in the admin transfer directory. Re-run `jobs_gen.sh` to sync it.

### `prod_01` instead of `prod_00`
NVFlare increments the `prod_XX` directory each time `nvflare provision` runs. All scripts dynamically detect the latest `prod_*` directory. Verify with:
```bash
ls workspace/iov_securefl_network/
```

### Data split paths wrong (`/home/hople/...`)
`jobs_gen.sh` auto-fixes stale hardcoded paths in data split JSONs before syncing to clients.

---

## Dependencies

| Package    | Version  | Purpose                    |
|------------|----------|----------------------------|
| nvflare    | ≥2.7.1   | Federated Learning framework |
| xgboost    | ≥3.2.0   | Gradient boosted trees      |
| scikit-learn | ≥1.5.0 | Metrics (Accuracy, F1, LogLoss) |
| pandas     | ≥2.2.0   | Data processing             |
| numpy      | ≥2.0,<2.1| Numerical operations        |
| uv         | latest   | Python environment manager  |
| Python     | 3.12.x   | Runtime                     |

---

## License

See [LICENSE](LICENSE).
