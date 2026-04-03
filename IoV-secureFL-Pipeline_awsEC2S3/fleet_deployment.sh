# REPLACE THESE WITH YOUR 5 VEHICLE PRIVATE IPs!
CORE_IPS="172.31.71.9 172.31.67.199 172.31.76.174 172.31.77.237 172.31.64.105"

for IP in $CORE_IPS; do
    echo "🚀 Bootstrapping Vehicle at $IP..."

    # 1. Use rsync to sync the project folder, explicitly excluding heavy/broken directories
    rsync -avz -e "ssh -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no" \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude '.ipynb_checkpoints' \
        --exclude '.git' \
        --exclude 'uv.lock' \
        --exclude 'data/raw' \
        --exclude 'workspace_iov_double_rf' \
        ~/IoV-secureFL-Pipeline_awsEC2S3/ ubuntu@$IP:~/IoV-secureFL-Pipeline_awsEC2S3/

    # 2. SSH in, force a clean uv install, and patch
    ssh -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF'
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        cd ~/IoV-secureFL-Pipeline_awsEC2S3
        uv venv --clear --python 3.12
        source .venv/bin/activate
        uv pip install nvflare xgboost scikit-learn pandas boto3
        python utils/patch_nvflare.py
EOF
done

echo "✅ Entire fleet is provisioned, synced, and patched!"
