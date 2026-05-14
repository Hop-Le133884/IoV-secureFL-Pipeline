#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# FLEET IP CONFIGURATION — update these after launching EC2
# ═══════════════════════════════════════════════════════════════
# After launching 6 EC2 instances (1 master + 5 clients), get
# the private IPv4 addresses from the AWS Console and fill them in.

# 5 vehicle client private IPs (site-1 through site-5):
CORE_IPS=(
    "<SITE1_PRIVATE_IP>"
    "<SITE2_PRIVATE_IP>"
    "<SITE3_PRIVATE_IP>"
    "<SITE4_PRIVATE_IP>"
    "<SITE5_PRIVATE_IP>"
)

# SSH key path (relative to this file's directory)
SSH_KEY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ec2Key/iov-dp-key.pem"
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no"
