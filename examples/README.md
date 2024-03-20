# Examples

Let's take `llemma-7b` as an example. We first need to prepare the model checkpoint in the `gpt-fast` format.

```bash
export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO=EleutherAI/llemma_7b

python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO \
    --target_precision bf16
```

Next, we can train the model with supervised fine-tuning.

```bash
export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

SFT_TRAIN_DATA=/path/to/your/sft/train/data.json
SFT_MODEL_SAVE_NAME=/path/to/your/sft/model/save/name

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 3 \
    --dataset "$SFT_TRAIN_DATA" \
    --dataset_format "prm-v2" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
    --resume_from_checkpoint
```

We also need to train the reward model.

```bash
export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

RM_DATA=/path/to/your/reward/model/data.json
RM_MODEL_SAVE_NAME=/path/to/your/reward/model/save/name

torchrun --standalone --nproc_per_node=4 \
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 2 \
    --dataset "$RM_DATA" \
    --dataset_format "prm-v3" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_on_every_token \
    --tensor_parallel_size 1 \
    --save_only_model True \
    --save_dir $DATA_DIR/checkpoints/$RM_MODEL_SAVE_NAME \
    --resume_from_checkpoint
```

Finally, we can run the policy model with PPO.

```bash
export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

LEARNING_RATE=2e-5
KL_COEF=0.01
EPOCH=50
NOPTEPOCHS=1
ROLLOUT_BATCH_SIZE=512
STEP_BATCH_SZIE=64
ROLLOUT_PER_DEVICE_BATCH_SIZE=32
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=8
STEP_PER_DEVICE_BATCH_SIZE=2

SAMPLING_TEMPARATURE=0.7

RM_MODEL_NAME=/path/to/your/reward/model/save/name
SFT_MODEL_NAME=/path/to/your/sft/model/save/name

PPO_TRAIN_DATA=/path/to/your/ppo/train/data.json
PPO_EVAL_DATA=/path/to/your/ppo/eval/data.json
PPO_TEST_DATA=/path/to/your/ppo/test/data.json
PPO_MODEL_SAVE_NAME=/path/to/your/ppo/model/save/name

torchrun --standalone --nproc_per_node=8 \
    finetune_ppo.py \
    --compile \
    --do_train \
    --base_checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --policy_checkpoint_path $DATA_DIR/checkpoints/$SFT_MODEL_NAME \
    --reward_checkpoint_path $DATA_DIR/checkpoints/$RM_MODEL_NAME \
    --source_max_len 384 \
    --target_max_len 768 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --noptepochs $NOPTEPOCHS \
    --ppo_warmup_steps 5 \
    --seed 42 \
    --dataset "$PPO_TRAIN_DATA" \
    --eval_dataset "$PPO_EVAL_DATA" \
    --test_dataset "$PPO_TEST_DATA" \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/$PPO_MODEL_SAVE_NAME \
    --resume_from_checkpoint False \
    --stop_token "\n\n# Answer\n\n" \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --whiten_rewards True \
    --temperature $SAMPLING_TEMPARATURE \
    --num_train_epochs $EPOCH \
    --policy_optimizer_cpu_offload True \
    --value_optimizer_cpu_offload True \
    --slow_cross_node_comm False \
    --vocab_parallel True \
    --sequence_parallel True \
    --policy_model_fsdp True \
    --value_model_fsdp True \
    --ref_policy_model_fsdp False \
    --ref_policy_model_bits 8 \
    --reward_model_fsdp False \
    --reward_model_bits 8 \
    --optim_dtype fp32 \
    --tensor_parallel_size 4 \
    --fsdp_consolidate_cpu_offload True \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --weight_decay 0.0 \
    --eval_steps 10 \
    --save_strategy "evaluate/accuracy_overall" \
    --save_steps 5 \
    --save_steps 10 \
    --save_total_limit 5 \
    --save_only_model True \
    --easy_outcome_reward True \
    --apply_process_reward True \
    --penalize_no_stop_token True \
    --relative_stop_token_penalty True \
    --penalty_reward_value -1.0 \
    --process_reward_upper_bound 0.5 \
    --process_reward_scale 1.0
```
