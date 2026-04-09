echo "Provisioning NVFlare Network..."
nvflare provision -p project.yml

PROD_DIR=$(ls -dt workspace/iov_securefl_network/prod_* 2>/dev/null | head -1)
echo "Distributing Vehicle Startup Kits from: $PROD_DIR"
CORE_IPS=("172.31.71.9" "172.31.67.199" "172.31.76.174" "172.31.77.237" "172.31.64.105")
SITE_NUM=1

for IP in "${CORE_IPS[@]}"; do
    echo "  -> Shipping site-${SITE_NUM} certificates to $IP..."
    ssh -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no ubuntu@$IP \
        "mkdir -p ~/IoV-secureFL-Pipeline_awsEC2/site-${SITE_NUM}"
    scp -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no -r \
        "${PROD_DIR}/site-${SITE_NUM}" ubuntu@$IP:~/IoV-secureFL-Pipeline_awsEC2/
    SITE_NUM=$((SITE_NUM + 1))
done

echo "✅ Network Provisioned and Certificates Distributed!"
