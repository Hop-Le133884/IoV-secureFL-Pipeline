#!/usr/bin/env bash

source .venv/bin/activate
MASTER_IP=$(hostname -I | awk '{print $1}')

CORE_IPS=("172.31.0.200" "172.31.0.28" "172.31.0.34" "172.31.0.21" "172.31.0.16")

echo "Generating project.yml for Master Server IP: $MASTER_IP"

cat << YML > project.yml
api_version: 3
name: iov_securefl_network
description: IoV Secure FL Network

participants:
  - name: server
    type: server
    org: aws_cluster
    fqdn: "$MASTER_IP"
    fed_learn_port: 8002
    admin_port: 8003
  - name: site-1
    type: client
    org: aws_cluster
  - name: site-2
    type: client
    org: aws_cluster
  - name: site-3
    type: client
    org: aws_cluster
  - name: site-4
    type: client
    org: aws_cluster
  - name: site-5
    type: client
    org: aws_cluster
  - name: admin@master.com
    type: admin
    org: aws_cluster
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
  - path: nvflare.lighter.impl.template.TemplateBuilder
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      overseer_agent:
        path: nvflare.ha.dummy_overseer_agent.DummyOverseerAgent
        overseer_exists: false
        args:
          sp_end_point: "$MASTER_IP:8002:8003"
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
YML

echo "Provisioning NVFlare Network..."
nvflare provision -p project.yml

# Detect the latest prod directory NVFlare just created
PROD_DIR=$(ls -dt workspace/iov_securefl_network/prod_* 2>/dev/null | head -1)
if [[ -z "$PROD_DIR" ]]; then
    echo "ERROR: No prod directory found after provisioning."
    exit 1
fi
echo "Using provisioned workspace: $PROD_DIR"

# Patch server start.sh to ensure 'server' hostname resolves locally
SERVER_START="$PROD_DIR/server/startup/start.sh"
if ! grep -q "etc/hosts" "$SERVER_START"; then
  sed -i 's|^\(if \[ \$doCloud\)|# Ensure '"'"'server'"'"' hostname resolves locally\nif ! grep -qE "^[0-9].*\\bserver\\b" /etc/hosts; then\n  echo "127.0.0.1 server" | sudo tee -a /etc/hosts > /dev/null\nfi\n\n\1|' "$SERVER_START"
  echo "Patched server start.sh with /etc/hosts fix"
fi

# Ensure 'server' hostname resolves on the master node (required by both
# the NVFlare server process and the fl_admin.sh admin client)
if ! grep -qE "^[0-9].*\bserver\b" /etc/hosts; then
    echo "$MASTER_IP server" | sudo tee -a /etc/hosts > /dev/null
    echo "Added '$MASTER_IP server' to /etc/hosts"
else
    echo "'server' already in /etc/hosts — skipping"
fi

echo "Distributing Vehicle Startup Kits..."
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
