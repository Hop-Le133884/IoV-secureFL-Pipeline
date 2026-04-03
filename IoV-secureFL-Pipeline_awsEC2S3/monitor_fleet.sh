#!/usr/bin/env bash
# Usage:
#   ./monitor_fleet.sh          → snapshot of latest results from all sites
#   ./monitor_fleet.sh live     → tail site-1 live (Ctrl+C to stop)
#   ./monitor_fleet.sh live 3   → tail site-3 live

KEY=ec2Key/iov-dp-key.pem
CORE_IPS=("172.31.71.9" "172.31.67.199" "172.31.76.174" "172.31.77.237" "172.31.64.105")

if [[ "${1}" == "live" ]]; then
    SITE_NUM="${2:-1}"
    IP="${CORE_IPS[$((SITE_NUM - 1))]}"
    echo "Tailing site-${SITE_NUM} ($IP) live — Ctrl+C to stop"
    ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$IP" \
        "tail -f ~/IoV-secureFL-Pipeline_awsEC2S3/client.log" 2>&1 \
        | grep --line-buffered -E "======>|ERROR|Stage [12]:|DP Stage|Macro-F1"
else
    SITE_NUM=1
    for IP in "${CORE_IPS[@]}"; do
        echo "=== site-${SITE_NUM} ($IP) ==="
        ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$IP" \
            "grep -E 'Stage [12]:|======>|Macro-F1|LogLoss|DP Stage' ~/IoV-secureFL-Pipeline_awsEC2S3/client.log | tail -6" 2>&1
        echo ""
        SITE_NUM=$((SITE_NUM + 1))
    done
fi
