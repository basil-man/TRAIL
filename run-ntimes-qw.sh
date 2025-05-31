#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=your_api_key
export RAY_memory_monitor_refresh_ms=0

BASE_RUN_NAME=${BASE_RUN_NAME:-"your run name"}
MODEL_BASE_PATH=your_model_path/Qwen2.5-3B-Instruct
DATA_DIR=data/math_qwen
LENGTH=1024
N_GPUS=2
TP=1
TOTAL_RUNS=4
MODEL_SIZE="3b"
MODEL_TYPE="qwen"
DATA_TYPE="test"

BETAS=(1.1 1.2 1.4)

for BETA in "${BETAS[@]}"; do
  echo "===== Running ANALYZER_HOST with --beta=${BETA} ====="
  CUDA_VISIBLE_DEVICES=1 python analyzer_host.py --beta "${BETA}" &
  ANALYZER_PID=$!
  echo "analyzer_host.py started (PID=${ANALYZER_PID})"

  if [[ "${BETA}" == "1.4" ]]; then
    MIN_REWARD=0.83
  elif [[ "${BETA}" == "1.1" ]]; then
    MIN_REWARD=0.93
  else
    MIN_REWARD=0.87
  fi

  for ((INDEX=1; INDEX<=TOTAL_RUNS; INDEX++)); do
      RUN_NAME="${BASE_RUN_NAME}-beta${BETA}-${INDEX}"
      echo ">>> Running experiment: $RUN_NAME"

      MODEL_DIR=checkpoints/${RUN_NAME}
      RESULT_FILE="all_results_${RUN_NAME}.csv"
      echo "step,run_name,benchmark,accuracy,avg_cot_length" > "$RESULT_FILE"

      python3 -m verl.trainer.main_ppo \
          algorithm.adv_estimator=grpo \
          +algorithm.use_replay_buffer=true \
          +algorithm.replay_buffer.capacity=500 \
          +algorithm.replay_buffer.top_percent=0.1 \
          +algorithm.replay_buffer.min_reward=${MIN_REWARD} \
          +algorithm.replay_buffer.min_distill_interval=1 \
          +algorithm.replay_buffer.max_distill_interval=20 \
          +algorithm.replay_buffer.fixed_distill_interval=10 \
          +algorithm.replay_buffer.distill_threshold_ratio=0.3 \
          +algorithm.replay_buffer.lambda_decay=2.5 \
          +algorithm.replay_buffer.off_policy_weight=0.1 \
          +algorithm.replay_buffer.batch_size=64 \
          +algorithm.replay_buffer.max_is_weight=10.0 \
          +algorithm.replay_buffer.microbatch_size=4 \
          data.train_files=${DATA_DIR}/train.parquet \
          data.val_files=${DATA_DIR}/test.parquet \
          data.train_batch_size=128 \
          data.val_batch_size=128 \
          data.max_prompt_length=512 \
          data.max_response_length=${LENGTH} \
          actor_rollout_ref.model.path=${MODEL_BASE_PATH} \
          actor_rollout_ref.actor.optim.lr=1e-6 \
          actor_rollout_ref.model.use_remove_padding=True \
          actor_rollout_ref.actor.ppo_mini_batch_size=32 \
          actor_rollout_ref.actor.ppo_micro_batch_size=16 \
          actor_rollout_ref.actor.use_kl_loss=True \
          actor_rollout_ref.actor.kl_loss_coef=0.001 \
          actor_rollout_ref.actor.kl_loss_type=low_var_kl \
          actor_rollout_ref.model.enable_gradient_checkpointing=True \
          actor_rollout_ref.model.lora_rank=8 \
          actor_rollout_ref.model.lora_alpha=16 \
          actor_rollout_ref.actor.fsdp_config.param_offload=False \
          actor_rollout_ref.actor.fsdp_config.grad_offload=False \
          actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
          actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
          actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
          actor_rollout_ref.rollout.name=vllm \
          actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
          actor_rollout_ref.rollout.n=4 \
          +actor_rollout_ref.rollout.disable_log_stats=False \
          actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
          reward_model.enable=False \
          algorithm.kl_ctrl.kl_coef=0.001 \
          trainer.critic_warmup=0 \
          trainer.val_before_train=False \
          trainer.default_local_dir=$MODEL_DIR \
          trainer.default_hdfs_dir=null \
          trainer.logger=['console','wandb'] \
          trainer.project_name='verl_math' \
          trainer.experiment_name=${RUN_NAME} \
          trainer.n_gpus_per_node=$N_GPUS \
          trainer.nnodes=1 \
          trainer.multisample_val=True \
          trainer.save_freq=20 \
          trainer.test_freq=1000000 \
          trainer.total_epochs=3 \
          trainer.num_keep_checkpoint=10 \
          trainer.resume_checkpoint=True

      for ((STEP=20; STEP<=160; STEP+=20)); do
          echo "Evaluating step $STEP for $RUN_NAME"

          python examples/model_merger.py \
              --local_dir checkpoints/${RUN_NAME}/actor/global_step_${STEP}/ \
              --save_dir ./model/${RUN_NAME}-${STEP}

          MODEL_PATH="./model/${RUN_NAME}-${STEP}"
          TOKENIZER_PATH=${MODEL_PATH}
          BASE_OUTPUT_DIR="outputs/${RUN_NAME}-${STEP}"

          for BENCHMARK in gsm8k math; do
              MAX_NEW_TOKENS=512
              [[ "$BENCHMARK" == "math" ]] && MAX_NEW_TOKENS=1024

              python eval_rl.py \
                  --model-path "$MODEL_PATH" \
                  --tokenizer-path "$TOKENIZER_PATH" \
                  --model-size "$MODEL_SIZE" \
                  --model-type "$MODEL_TYPE" \
                  --benchmark "$BENCHMARK" \
                  --data-type "$DATA_TYPE" \
                  --output-dir "${BASE_OUTPUT_DIR}/step_${STEP}/${BENCHMARK}/" \
                  --max_new_tokens "$MAX_NEW_TOKENS" \
                  --temperature 0.0 \
                  --eval_batch_size 16 \
                  --seed 42

              METRICS_FILE="${BASE_OUTPUT_DIR}/step_${STEP}/${BENCHMARK}/${MODEL_SIZE}/Original/test/samples/metrics.json"
              if [ -f "$METRICS_FILE" ]; then
                  ACCURACY=$(jq '.accuracy' "$METRICS_FILE")
                  AVG_LEN=$(jq '.avg_cot_length' "$METRICS_FILE")
                  echo "${STEP},${RUN_NAME},${BENCHMARK},${ACCURACY},${AVG_LEN}" >> "$RESULT_FILE"
              else
                  echo "${STEP},${RUN_NAME},${BENCHMARK},ERROR,ERROR" >> "$RESULT_FILE"
              fi
          done
      done

      echo "Cleaning up checkpoints for $RUN_NAME"
      #rm -rf "checkpoints/${RUN_NAME}"
  done

  echo "Stopping analyzer_host.py (PID=${ANALYZER_PID})"
  kill "${ANALYZER_PID}"
  wait "${ANALYZER_PID}" 2>/dev/null || true
done

echo "All experiments done."
