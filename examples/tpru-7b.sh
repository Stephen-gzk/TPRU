#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=./model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=./examples/tpru-7b.yaml \
    data.train_files=./data/tpru25k@train \
    data.val_files=./data/tpru25k@test \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=7 \
    trainer.experiment_name=tpru-7b \
    trainer.n_gpus_per_node=8

