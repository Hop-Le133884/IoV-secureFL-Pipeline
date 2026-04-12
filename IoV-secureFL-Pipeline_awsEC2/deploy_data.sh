#!/usr/bin/env bash
# Lightweight per-seed data deploy — copies only the new training CSVs to clients.
# Run this after data_split_gen.sh regenerates splits for a new seed.
# Does NOT reinstall packages or touch certificates.
#
# Usage:  bash deploy_data.sh

CORE_IPS=("172.31.0.97" "172.31.0.242" "172.31.0.73" "172.31.0.69" "172.31.0.9")

REPO_ROOT="$(realpath "$(dirname "$0")")"
SSH_KEY="${REPO_ROOT}/ec2Key/iov-dp-key.pem"
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no"

echo "Deploying updated training CSVs to fleet..."
SITE_NUM=1
for IP in "${CORE_IPS[@]}"; do
    scp ${SSH_OPTS} \
        "${REPO_ROOT}/data/processed/vehicle_site-${SITE_NUM}_train.csv" \
        ubuntu@${IP}:~/IoV-secureFL-Pipeline_awsEC2/data/processed/
    echo "  -> vehicle_site-${SITE_NUM}_train.csv → ${IP}"
    SITE_NUM=$((SITE_NUM + 1))
done

echo "Data deployed."
