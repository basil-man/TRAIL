#!/bin/bash
RUN_NAME="Qwen2.5-0.5B-Instruct-512-mean-repleybuffer-5"
if [ ! -f all_results_$RUN_NAME.csv ]; then
    echo "step,benchmark,accuracy,avg_cot_length" > all_results_$RUN_NAME.csv
fi


for ((STEP=20; STEP<=160; STEP+=20))
do
    echo "==============================="
    echo "Running with step = $STEP"
    echo "==============================="

    
    python examples/model_merger_single_gpu.py     --local_dir checkpoints/${RUN_NAME}/actor/global_step_${STEP}/     --save_dir ./model/${RUN_NAME}-${STEP}


    MODEL="${RUN_NAME}-${STEP}"
    MODEL_PATH="./model/${MODEL}"
    TOKENIZER_PATH=${MODEL_PATH}

    BASE_OUTPUT_DIR="outputs/${MODEL}"
    DATA_TYPE="test"
    MODEL_SIZE="0.5b"
    MODEL_TYPE="qwen"
    
    python eval_rl.py \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --model-size "$MODEL_SIZE" \
        --model-type "$MODEL_TYPE" \
        --benchmark "gsm8k" \
        --data-type "$DATA_TYPE" \
        --output-dir "${BASE_OUTPUT_DIR}/step_${STEP}/gsm8k/" \
        --max_new_tokens 512 \
        --temperature 0.0 \
        --eval_batch_size 16 \
        --seed 42
    python eval_rl.py \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --model-size "$MODEL_SIZE" \
        --model-type "$MODEL_TYPE" \
        --benchmark "math" \
        --data-type "$DATA_TYPE" \
        --output-dir "${BASE_OUTPUT_DIR}/step_${STEP}/math/" \
        --max_new_tokens 1024 \
        --temperature 0.0 \
        --eval_batch_size 16 \
        --seed 42
    
    for BENCHMARK in gsm8k math
    do
        METRICS_FILE="./outputs/${MODEL}/step_${STEP}/${BENCHMARK}/${MODEL_SIZE}/Original/test/samples/metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            ACCURACY=$(jq '.accuracy' "$METRICS_FILE")
            AVG_LEN=$(jq '.avg_cot_length' "$METRICS_FILE")
            echo "${STEP},${BENCHMARK},${ACCURACY},${AVG_LEN}" >> all_results_$RUN_NAME.csv
        else
            echo "${STEP},${BENCHMARK},ERROR,ERROR" >> all_results_$RUN_NAME.csv
        fi
    done


    rm -r ./model/${RUN_NAME}-${STEP}
done
