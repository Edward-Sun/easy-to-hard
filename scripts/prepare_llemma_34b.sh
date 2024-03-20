set -e
set -x

export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO=EleutherAI/llemma_34b

python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO \
    --target_precision bf16
