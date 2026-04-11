#!/usr/bin/env bash
# Wipes EC2 client nodes back to first-time state.
# After this, run fleet_deployment.sh to re-bootstrap before start_fleet.sh.
# Usage: ./clean_fleet.sh

KEY=ec2Key/iov-dp-key.pem

CORE_IPS=("172.31.0.97" "172.31.0.242" "172.31.0.73" "172.31.0.69" "172.31.0.9")


SITE_NUM=1

echo "Cleaning all client nodes to first-time state..."
for IP in "${CORE_IPS[@]}"; do
    echo "  -> Cleaning site-${SITE_NUM} ($IP)..."
    ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$IP" bash <<'REMOTE'
        REPO=~/IoV-secureFL-Pipeline_awsEC2

        # Kill any running NVFlare / sub_start processes
        pkill -f nvflare      2>/dev/null
        pkill -f sub_start    2>/dev/null
        pkill -f server_train 2>/dev/null
        sleep 1

        # Wipe entire project repo
        rm -rf "$REPO"

        echo "done"
REMOTE
    SITE_NUM=$((SITE_NUM + 1))
done

echo ""
echo "All client nodes wiped."
echo "Next step: ./fleet_deployment.sh  (re-bootstraps venv + syncs project)"
