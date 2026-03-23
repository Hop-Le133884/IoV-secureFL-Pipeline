#!/usr/bin/env bash
# Update this path to your specific IoV CSV location
DATASET_PATH="${1}/CICIoV2024.csv"
OUTPUT_PATH="${1}/IoV/data_splits"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Error: CICIoV2024.csv not found in ${DATASET_PATH}"
    exit 1
fi

echo "Generating Stratified IoV data splits from ${DATASET_PATH}..."

# We focus on 5 sites for your research
for site_num in 5; 
do  
    # Stratification ensures each vehicle sees all 6 attack classes
    python3 utils/prepare_data_split.py \
    --data_path "${DATASET_PATH}" \
    --site_num ${site_num} \
    --size_valid 140822 \
    --out_path "${OUTPUT_PATH}"
done

echo "Stratified splits generated in ${OUTPUT_PATH}"