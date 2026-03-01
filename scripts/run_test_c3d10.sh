#!/bin/bash
cd /home/nishioka/Payload2026
python src/run_batch.py \
    --doe doe_test_c3d10.json \
    --output_dir dataset_test_c3d10 \
    --gen_script realistic \
    --global_seed 25 \
    --defect_seed 10 \
    --work_dir abaqus_work_test \
    --n_cpus 8 \
    --memory 40gb \
    --keep_inp \
    --force
