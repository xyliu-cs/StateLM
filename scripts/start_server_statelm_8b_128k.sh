set -e
set -u


model_name="StateLM-8B-128k"
model_path="/path/to/your/model"

command="vllm serve $model_path \
        --tensor-parallel-size 8 \
        --dtype float16 \
        --max_model_len 131072 \
        --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
        --disable-log-requests \
        --served-model-name $model_name \
        --gpu-memory-utilization 0.8 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --reasoning-parser deepseek_r1 \
        --chat-template ./qwen3.jinja "

num_gpus=$(nvidia-smi --list-gpus | wc -l)

declare -A port_idx
port_idx=(
    ["8080"]="0,1,2,3,4,5,6,7"
)

conda_env=/path/to/the/vllm/environment
# Launch a tmux session for each port and GPU pair
for port in ${!port_idx[@]}; do
    command_cuda="export CUDA_VISIBLE_DEVICES=${port_idx[$port]}"
    command_port="$command --port $port"
    tmux new -d -s $port
    tmux send-keys -t $port "conda activate ${conda_env}" ENTER
    tmux send-keys -t $port "$command_cuda" ENTER
    tmux send-keys -t $port "$command_port" ENTER
done

for port in ${!port_idx[@]}; do
    echo "$model_name server started at port $port"
done
