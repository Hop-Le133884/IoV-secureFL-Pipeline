#!/usr/bin/env bash

# The job name is now standardized for your IoV research
JOB_NAME="iov_double_rf_5_sites"
WORKSPACE="workspace_iov_double_rf"

echo "Starting NVFLARE Simulator for Double RF Research..."

nvflare simulator jobs/${JOB_NAME} \
-w ${WORKSPACE} \
-n 5 \
-t 5