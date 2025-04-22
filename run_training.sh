#!/bin/bash
# run_training.sh
# Script to run the three training configurations for TAPMIC

# Make sure we're in the project directory
cd /Users/param/Desktop/TAPMIC

# Create directory for logs
mkdir -p logs

# Set common parameters
MODEL_NAME="roberta-base"
BATCH_SIZE=16
MAX_LENGTH=128
EPOCHS=5
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
DROPOUT_RATE=0.1
GRAD_ACCUM_STEPS=2
EARLY_STOPPING=3
SEED=42

# Set date for run names
DATE=$(date +%Y%m%d)

echo "========================================="
echo "Starting TAPMIC training runs: $DATE"
echo "========================================="

# Run 1: RoBERTa with text features only (baseline)
echo "Run 1: RoBERTa with text features only (baseline)"
echo "----------------------------------------"
python TAPMIC/models/run_roberta.py \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --dropout_rate $DROPOUT_RATE \
  --grad_accumulation_steps $GRAD_ACCUM_STEPS \
  --early_stopping_patience $EARLY_STOPPING \
  --run_name "${DATE}_text_only" \
  --seed $SEED \
  --fp16 \
  --wandb 2>&1 | tee logs/text_only_${DATE}.log

# Run 2: RoBERTa with text + temporal features
echo "Run 2: RoBERTa with text + temporal features"
echo "----------------------------------------"
python TAPMIC/models/run_roberta.py \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --dropout_rate $DROPOUT_RATE \
  --grad_accumulation_steps $GRAD_ACCUM_STEPS \
  --early_stopping_patience $EARLY_STOPPING \
  --use_temporal \
  --run_name "${DATE}_text_temporal" \
  --seed $SEED \
  --fp16 \
  --wandb 2>&1 | tee logs/text_temporal_${DATE}.log

# Run 3: RoBERTa with text + temporal + speaker credibility features
echo "Run 3: RoBERTa with text + temporal + speaker credibility features"
echo "----------------------------------------"
python TAPMIC/models/run_roberta.py \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --dropout_rate $DROPOUT_RATE \
  --grad_accumulation_steps $GRAD_ACCUM_STEPS \
  --early_stopping_patience $EARLY_STOPPING \
  --use_temporal \
  --use_credibility \
  --run_name "${DATE}_text_temporal_credibility" \
  --seed $SEED \
  --fp16 \
  --wandb 2>&1 | tee logs/text_temporal_credibility_${DATE}.log

echo "========================================="
echo "All training runs completed"
echo "========================================="

# Run evaluation comparison
echo "Running model comparison..."
python TAPMIC/models/model_comparison.py \
  --baseline_model checkpoints/${DATE}_text_only_best.pt \
  --temporal_model checkpoints/${DATE}_text_temporal_best.pt \
  --full_model checkpoints/${DATE}_text_temporal_credibility_best.pt \
  --run_name "${DATE}_comparison" \
  --wandb 2>&1 | tee logs/comparison_${DATE}.log

echo "========================================="
echo "TAPMIC training and comparison completed"
echo "=========================================" 