#!/usr/bin/env bash

source .venv/bin/activate

SERVER_IP=$(hostname -I | awk '{print $1}')

CORE_IPS=("172.31.0.200" "172.31.0.28" "172.31.0.34" "172.31.0.21" "172.31.0.16")

SITE_NUM=1

echo "Starting 5 Vehicle Clients..."
for IP in "${CORE_IPS[@]}"; do
    echo "  -> Igniting site-${SITE_NUM} at $IP..."
    ssh -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no ubuntu@$IP "
      grep -q 'server' /etc/hosts || echo '${SERVER_IP} server' | sudo tee -a /etc/hosts > /dev/null
      cd ~/IoV-secureFL-Pipeline_awsEC2 && source .venv/bin/activate && nohup ./site-${SITE_NUM}/startup/start.sh > client.log 2>&1 &
    "
    SITE_NUM=$((SITE_NUM + 1))
done

echo "✅ Fleet is fully online and training-ready!"
