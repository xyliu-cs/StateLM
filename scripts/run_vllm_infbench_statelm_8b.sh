#!/usr/bin/env bash
set -e
set -u

export ES_HOST="https://localhost:9200"
export ES_USER="elastic"
export ES_PASS="your_password"
export ES_CA_CERT="http_ca.crt"

PROJECT_BASE="/path/to/your/project/base"
cd "$PROJECT_BASE" || { echo "Failed to cd to PROJECT_BASE"; exit 1; }


SAVE_DIR="/path/to/your/save/directory"

RUN_ID="StateLM-8B"
OPENAI_FILE="openai_endpoint.json"

DATASET=lindsay21/InfiniteBench
SPLITS=("longbook_choice_eng")
RESULT_TXT=eval_results/${RUN_ID}_infbench_choice_results.txt

TRAJECTORIES_DIR="${SAVE_DIR}/trajectories/${DATASET}/${RUN_ID}"
RESULTS_DIR="${SAVE_DIR}/results/${DATASET}/${RUN_ID}"

TEMP=0.7
TOP_P=0.8
TOP_K=20
MAX_CONTEXT=32000
MAX_CONTEXT_EXP=32000
MAX_TURNS_EXP=150
MAX_TURNS_TO_FAIL=200
MAX_OUTPUT_TOKENS=2048

# prompts are stored in tools_and_prompt/prompts.py
SYSTEM_PROMPT_NAME="STATELM_SYSTEM_PROMPT"

TOOL_CONFIG_PATH="LC_Agent/tools_and_prompt/statelm_tools_optimized.json"

for i in {1..5}; do
    for SPLIT in ${SPLITS[@]}; do
        OUTPUT_FP=${RESULTS_DIR}/${SPLIT}_generations_$(date +%Y%m%d_%H%M%S).jsonl
        TRAJECTORIES_SAVE_DIR=${TRAJECTORIES_DIR}/$(date +%Y%m%d_%H%M%S)
        RESULTS_SAVE_DIR=${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)
        python -m LC_Agent.scripts.hf_test_runner eval_hfds_statelm \
            --vllm_cfg $OPENAI_FILE \
            --temperature $TEMP \
            --top_p $TOP_P \
            --top_k $TOP_K \
            --max_turns_exp $MAX_TURNS_EXP \
            --max_context_exp $MAX_CONTEXT_EXP \
            --max_context $MAX_CONTEXT \
            --max_output_tokens $MAX_OUTPUT_TOKENS \
            --tool_config_path $TOOL_CONFIG_PATH \
            --system_prompt_name $SYSTEM_PROMPT_NAME \
            --dataset_name $DATASET \
            --dataset_split $SPLIT \
            --item_to_question LC_Agent/inference/hf_process_fns.py:infinitebench_${SPLIT}_i2q \
            --item_to_context LC_Agent/inference/hf_process_fns.py:infinitebench_${SPLIT}_i2c \
            --item_to_answer  LC_Agent/inference/hf_process_fns.py:infinitebench_${SPLIT}_i2a \
            --trajectory_dir $TRAJECTORIES_SAVE_DIR \
            --result_dir $RESULTS_SAVE_DIR \
            --output_fp $OUTPUT_FP \
            --tokenizer_path Qwen/Qwen3-8B \
            --max_turns_to_fail $MAX_TURNS_TO_FAIL \
            --n_proc 1 \
            --resume False \
            --version v4opt

        python LC_Agent/inference/compute_scores.py compute_scores \
            --preds_path "$OUTPUT_FP" \
            --task_name $SPLIT \
            --model_name $RUN_ID \
            --label_key "correct_answer" \
            --pred_key "final_answer" \
            --results_output $RESULT_TXT

        python LC_Agent/inference/compute_scores.py evaluate_choice_file \
            --file_path "$OUTPUT_FP" \
            --label_key "correct_answer" \
            --pred_key "final_answer"

    done
done