#!/usr/bin/env bash
# Bootstraps each client EC2 node with only the files it needs to run FL.
#
# Files copied per site:
#   utils/patch_nvflare.py              — one-time NVFlare patch (all sites)
#   data/processed/vehicle_site-N_train.csv — this site's training data only
#
# Files NOT copied (handled by other scripts):
#   site-N/startup/  — network_provision.sh (certs + start.sh)
#   data/IoV/data_splits/data_site-N.json — jobs_gen.sh
#   iov_executor.py, iov_data_loader.py  — sent by NVFlare at job runtime

CORE_IPS=("172.31.0.97" "172.31.0.242" "172.31.0.73" "172.31.0.69" "172.31.0.9")

REPO_ROOT="$(realpath "$(dirname "$0")")"
SSH_KEY="${REPO_ROOT}/ec2Key/iov-dp-key.pem"
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no"

SITE_NUM=1
for IP in "${CORE_IPS[@]}"; do
    echo "Bootstrapping site-${SITE_NUM} at ${IP}..."

    # 1. Create required directories
    ssh ${SSH_OPTS} ubuntu@${IP} \
        "mkdir -p ~/IoV-secureFL-Pipeline_awsEC2/{utils,data/processed}"

    # 2. Copy the NVFlare patch script (needed once to fix NVFlare installation)
    scp ${SSH_OPTS} \
        "${REPO_ROOT}/utils/patch_nvflare.py" \
        ubuntu@${IP}:~/IoV-secureFL-Pipeline_awsEC2/utils/

    # 3. Copy only this site's training CSV (not all 5)
    scp ${SSH_OPTS} \
        "${REPO_ROOT}/data/processed/vehicle_site-${SITE_NUM}_train.csv" \
        ubuntu@${IP}:~/IoV-secureFL-Pipeline_awsEC2/data/processed/
    echo "  -> Copied vehicle_site-${SITE_NUM}_train.csv to ${IP}"

    # 4. Install Python env and patch NVFlare
    ssh ${SSH_OPTS} ubuntu@${IP} << 'EOF'
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        cd ~/IoV-secureFL-Pipeline_awsEC2
        uv venv --clear --python 3.12
        source .venv/bin/activate
        uv pip install nvflare xgboost scikit-learn pandas
        python utils/patch_nvflare.py
EOF
    echo "  -> site-${SITE_NUM} bootstrapped"
    SITE_NUM=$((SITE_NUM + 1))
done

echo "Fleet provisioned with minimal required files!"
