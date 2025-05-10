#!/bin/bash

# =========[通用配置]=========
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=bf9ae74795eb47b66c791ae2a3952ca1eacacf12
export RAY_memory_monitor_refresh_ms=0

# 更新基础运行名称以反映固定间隔策略
BASE_RUN_NAME="DeepSeek-R1-Distill-Qwen-1.5B-fixed10-replay-buffer-0.95limit-top10"
MODEL_BASE_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
DATA_DIR=data/gsm8k
LENGTH=2048
N_GPUS=2
TP=1
TOTAL_RUNS=1
MODEL_SIZE="1.5b"
MODEL_TYPE="deepseek"
DATA_TYPE="test"

# =========[主循环：多次实验，RUN_NAME自增]=========
for ((INDEX=1; INDEX<=TOTAL_RUNS; INDEX++)); do
    RUN_NAME="${BASE_RUN_NAME}-${INDEX}"
    echo ">>> Running experiment: $RUN_NAME"

    MODEL_DIR=checkpoints/${RUN_NAME}
    RESULT_FILE="all_results_${RUN_NAME}.csv"
    echo "step,run_name,benchmark,accuracy,avg_cot_length" > "$RESULT_FILE"

    # ===[1. 启动训练]===
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        +algorithm.use_replay_buffer=true \
        +algorithm.replay_buffer.capacity=500 \
        +algorithm.replay_buffer.top_percent=0.1 \
        +algorithm.replay_buffer.min_reward=0.95 \
        +algorithm.replay_buffer.min_distill_interval=5 \
        +algorithm.replay_buffer.max_distill_interval=20 \
        +algorithm.replay_buffer.fixed_distill_interval=10 \
        +algorithm.replay_buffer.distill_threshold_ratio=0.3 \
        +algorithm.replay_buffer.lambda_decay=2.5 \
        +algorithm.replay_buffer.off_policy_weight=0.1 \
        +algorithm.replay_buffer.batch_size=64 \
        +algorithm.replay_buffer.max_is_weight=10.0 \
        +algorithm.replay_buffer.microbatch_size=2 \
        data.train_files=${DATA_DIR}/train.parquet \
        data.val_files=${DATA_DIR}/test.parquet \
        data.train_batch_size=128 \
        data.val_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=${LENGTH} \
        actor_rollout_ref.model.path=${MODEL_BASE_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=4 \
        +actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
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

    # ===[2. 评估每个 step]===
    for ((STEP=20; STEP<=160; STEP+=20)); do
        echo "Evaluating step $STEP for $RUN_NAME"

        python examples/model_merger_single_gpu.py \
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

        #rm -r ./model/${RUN_NAME}-${STEP}
    done

    # ===[3. 删除 checkpoints 释放空间]===
    echo "Cleaning up checkpoints for $RUN_NAME"
    #rm -rf "checkpoints/${RUN_NAME}"

done