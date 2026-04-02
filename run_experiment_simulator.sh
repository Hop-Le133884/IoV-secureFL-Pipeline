#!/usr/bin/env bash

# Override via env vars (used by run_dp_sweep.sh for parallel runs):
#   JOB_NAME=iov_double_rf_5_sites_eps100_seed42 bash run_experiment_simulator.sh
#   WORKSPACE=workspace_iov_double_rf_eps100_seed42 bash run_experiment_simulator.sh
JOB_NAME="${JOB_NAME:-iov_double_rf_5_sites}"
WORKSPACE="${WORKSPACE:-workspace_iov_double_rf}"

echo "Starting NVFLARE Simulator for Double RF Research..."
echo "  Job:       ${JOB_NAME}"
echo "  Workspace: ${WORKSPACE}"

nvflare simulator jobs/${JOB_NAME} \
-w ${WORKSPACE} \
-n 5 \
-t 5