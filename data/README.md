# Data

## PRM800K

Prepare the [PRM800K](https://arxiv.org/abs/2305.20050) SFT training datasets

```bash
mkdir -p downloads/prm800k_data
cd downloads/prm800k_data
wget https://github.com/openai/prm800k/raw/main/prm800k/data/phase1_test.jsonl
wget https://github.com/openai/prm800k/raw/main/prm800k/data/phase1_train.jsonl
wget https://github.com/openai/prm800k/raw/main/prm800k/data/phase2_test.jsonl
wget https://github.com/openai/prm800k/raw/main/prm800k/data/phase2_train.jsonl
cd ../..

mkdir -p downloads/math_splits
cd downloads/math_splits
wget https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl
wget https://github.com/openai/prm800k/raw/main/prm800k/math_splits/train.jsonl
cd ../..
```

```bash
# level 1-3
export DATA_DIR=/path/to/your/data/dir

python -u prepare_prm800k_sft.py \
    --prm_data_pattern "downloads/prm800k_data/*train.jsonl" \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3" \
    --output_path $DATA_DIR/train_sft_1to3_prm800k.json
```

```bash
# level 1-5
export DATA_DIR=/path/to/your/data/dir

python -u prepare_prm800k_sft.py \
    --prm_data_pattern "downloads/prm800k_data/*train.jsonl" \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --output_path $DATA_DIR/train_sft_1to5_prm800k.json
```

Prepare the [PRM800K](https://arxiv.org/abs/2305.20050) PRM training datasets

```bash
# level 1-3
export DATA_DIR=/path/to/your/data/dir

python -u prepare_prm800k_rm.py \
    --prm_data_pattern "downloads/prm800k_data/*train.jsonl" \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3" \
    --output_path $DATA_DIR/train_prm_1to3_prm800k.json
```

```bash
# level 1-5
export DATA_DIR=/path/to/your/data/dir

python -u prepare_prm800k_rm.py \
    --prm_data_pattern "downloads/prm800k_data/*train.jsonl" \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --output_path $DATA_DIR/train_prm_1to5_prm800k.json
```

## MetaMath (Math-Shepherd)

Prepare the [MetaMath](https://arxiv.org/abs/2309.12284) SFT training datasets for Hendrycks's MATH:

```bash
# level 1-3
export DATA_DIR=/path/to/your/data/dir

python -u prepare_metamath.py \
    --levels "Level 1, Level 2, Level 3" \
    --pruned_numbers 4 \
    --epoch 3 \
    --pruned_output_path $DATA_DIR/train_sft_1to3_metamath_pruned_epoch_3.json \
    --output_path $DATA_DIR/train_sft_1to3_metamath.json
```

```bash
# level 1-5
export DATA_DIR=/path/to/your/data/dir

python -u prepare_metamath.py \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --pruned_numbers 4 \
    --epoch 3 \
    --pruned_output_path $DATA_DIR/train_sft_1to5_metamath_pruned_epoch_3.json \
    --output_path $DATA_DIR/train_sft_1to5_metamath.json
```

Prepare the [Math-Shepherd](https://arxiv.org/abs/2312.08935) PRM training dataset for Hendrycks's MATH:

```bash
export DATA_DIR=/path/to/your/data/dir

python -u prepare_metamath_shepherd.py \
    --output_1to3_path $DATA_DIR/train_prm_math_shepherd_level1-3.json \
    --output_1to5_path $DATA_DIR/train_prm_math_shepherd_level1-5.json
```

## PPO Data

Prepare the PPO training and validation datasets

```bash
# level 1-3
export DATA_DIR=/path/to/your/data/dir

python -u prepare_ppo_train_valid.py \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3" \
    --train_output_path $DATA_DIR/train_ppo_1to3.json \
    --valid_output_path $DATA_DIR/valid_ppo_1to3.json.tmp \ # we will not use this file
    --seed 1234
```

```bash
# level 1-5
export DATA_DIR=/path/to/your/data/dir

python -u prepare_ppo_train_valid.py \
    --train_math_path downloads/math_splits/train.jsonl \
    --test_math_path downloads/math_splits/test.jsonl \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --train_output_path $DATA_DIR/train_ppo_1to5.json \
    --valid_output_path $DATA_DIR/valid_ppo_1to5.json \
    --seed 1234
```

Prepare the MATH500 test dataset

```bash
python -u prepare_ppo_test.py \
    --test_math_path downloads/math_splits/test.jsonl \
    --output_path $DATA_DIR/test_ppo.json
```
