#!/usr/bin/env bash
set -e
set -u

export ES_HOST="https://localhost:9200"
export ES_USER="elastic"
export ES_PASS="your_password"
export ES_CA_CERT="/path/to/http_ca.crt"

PROJECT_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_BASE" || { echo "Failed to cd to PROJECT_BASE"; exit 1; }

SAVE_DIR="/path/to/your/save/directory"
RUN_ID="StateLM-8B-128K"
OPENAI_FILE="StateLM/openai_endpoint_example.json"

DATASET=lindsay21/niah-single
SPLITS=(
    "32768"
    "65536"
    "131072"
    "262144"
    "524288"
    "786432"
    "1048576"
    "2097152"
)
BENCHMARK="synthetic"

TRAJECTORIES_DIR="${SAVE_DIR}/trajectories/${DATASET}/${RUN_ID}"
RESULTS_DIR="${SAVE_DIR}/results/${DATASET}/${RUN_ID}"

TEMP=0.7
TOP_P=0.8
TOP_K=20
MAX_CONTEXT=32000
MAX_CONTEXT_EXP=32000
MAX_TURNS_EXP=500
MAX_TURNS_TO_FAIL=999
MAX_OUTPUT_TOKENS=1024


# prompts are stored in tools_and_prompt/prompts.py
SYSTEM_PROMPT_NAME="STATELM_SYSTEM_PROMPT"
TOOL_CONFIG_PATH="StateLM/tools_and_prompt/statelm_tools_without_search.json"

for i in {1..5}; do
  TRAJECTORIES_TIME=${TRAJECTORIES_DIR}/$(date +%Y%m%d_%H%M%S)
  RESULTS_TIME=${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)
  for SPLIT in ${SPLITS[@]}; do
    TRAJECTORIES_SAVE_DIR=${TRAJECTORIES_TIME}/${SPLIT}
    RESULTS_SAVE_DIR=${RESULTS_TIME}/${SPLIT}
    OUTPUT_DIR=${RESULTS_DIR}/${SPLIT}_$(date +%Y%m%d_%H%M%S)
    OUTPUT_FP=${OUTPUT_DIR}/${SPLIT}_generations.jsonl
    python -m StateLM.scripts.hf_test_runner eval_hfds_statelm \
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
        --item_to_question StateLM/inference/hf_process_fns.py:ruler_niah_i2q \
        --item_to_context StateLM/inference/hf_process_fns.py:ruler_niah_i2c \
        --item_to_answer  StateLM/inference/hf_process_fns.py:ruler_niah_i2a \
        --item_to_meta StateLM/inference/hf_process_fns.py:ruler_niah_i2meta \
        --output_postprocess StateLM/inference/hf_process_fns.py:ruler_niah_postprocess \
        --correct_answer_key "outputs" \
        --model_answer_key "pred" \
        --trajectory_dir $TRAJECTORIES_SAVE_DIR \
        --result_dir $RESULTS_SAVE_DIR \
        --output_fp $OUTPUT_FP \
        --tokenizer_path Qwen/Qwen3-8B \
        --max_turns_to_fail $MAX_TURNS_TO_FAIL \
        --n_proc 1 \
        --resume False \
        --version niah

    cd StateLM/inference
    python evaluate.py \
        --data_dir "${OUTPUT_DIR}" \
        --benchmark "${BENCHMARK}" \
        --results_dir "${PROJECT_BASE}/eval_results/${RUN_ID}"
    cd -
    done
done