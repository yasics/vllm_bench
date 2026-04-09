      
#!/bin/bash


### 安装依赖
# 安装依赖过程不能出错，否则自动退出
set -e  
if ! command -v numactl > /dev/null 2>&1; then
    apt-get update && apt-get install -y numactl
fi
set +e


# 设置统一的目标文件夹，用于存放结果
DATE=`date +%m%d%H%M%S`
LOG_DIR="results_${DATE}/"
mkdir -p ${LOG_DIR}
cp $0 ${LOG_DIR}

# 将本脚本的执行输出同时重定向到文件中
exec > >(tee ${LOG_DIR}run.log) 2>&1

# 导入清理vllm服务函数
source cleanup.sh

declare -A resolution_configs=(
    ["720p"]='{(1280, 720, 1): 1.0}'
    ["1080p"]='{(1920, 1080, 1): 1.0}'
    ["4k"]='{(3840, 2160, 1): 1.0}'
)

### 定义确认服务是否可以访问的函数
function ping_server(){
    local addr=$1
    local port=$2
    local pid=$3

    url="http://${addr}:${port}"
    timeout=1200
    start_time=$(date +%s)

    echo "Started api_server with pid ${pid} at ${url}"
    echo "Waiting for $url can be connected..."
    while true; do
        
        if curl -sSf --max-time 1 "$url/health" &>/dev/null; then
            echo "✅ $url is reachable"
            return 0
        else
            if ! kill -0 $pid 2>/dev/null; then
                echo "❌ $url is unreacheable and process $pid has been terminated"
                return 2
            fi
        fi

        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        if ((elapsed_time >= timeout)); then
            echo "❌ $url is unreachable after $timeout seconds"
            return 1
        fi

        sleep 10
    done
}

### 定义client端函数
function run_client() {
    local model=$1
    local tp_size=$2
    local pp_size=$3
    local concurrency=$4
    local input_len=$5
    local output_len=$6
    local image_num=$7
    local resolution_name=$8
    local client_log_dir=$(realpath $9)
    local image_resolution="${resolution_configs[$resolution_name]}"
    

    local first_run_log_dir="${client_log_dir}/first_run"
    local second_run_log_dir="${client_log_dir}/second_run"
    mkdir -p ${first_run_log_dir}
    mkdir -p ${second_run_log_dir}

    # 发起测试
    chip_num=$((tp_size*pp_size))
    num_prompts=$((concurrency*10))
    
    zero_client_cmds="vllm bench serve --backend openai-chat --model ${model} --trust-remote-code --endpoint /v1/chat/completions --ignore-eos --num-prompts 1  --dataset-name random-mm --random-input-len ${input_len} --random-output-len ${output_len} --metadata tp=${tp_size} pp=${pp_size} test_no=first gpu_num=$((chip_num)) --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0 --random-mm-limit-mm-per-prompt '{\"image\": 16, \"video\": 0}' --random-mm-bucket-config '${image_resolution}'"
    eval "$zero_client_cmds"

    base_client_cmds="vllm bench serve --backend openai-chat --model ${model} --trust-remote-code --endpoint /v1/chat/completions --ignore-eos --num-prompts ${num_prompts} --max-concurrency ${concurrency} --dataset-name random-mm --random-input-len ${input_len} --random-output-len ${output_len}"

    echo "-------------------------------------"
    echo "正在执行第一次测试"
    first_client_cmds="${base_client_cmds} --metadata tp=${tp_size} pp=${pp_size} test_no=first gpu_num=$((chip_num)) input_len=${input_len} output_len=${output_len} --random-mm-base-items-per-request ${image_num} --random-mm-num-mm-items-range-ratio 0 --random-mm-limit-mm-per-prompt '{\"image\": 16, \"video\": 0}' --random-mm-bucket-config '${image_resolution}' --save-result --result-dir ${first_run_log_dir}"
    echo "$first_client_cmds"
    eval "$first_client_cmds"
    
    echo "-------------------------------------"
}


### 定义测试函数
function run(){
    local model=$1
    local tp_size=$2
    local pp_size=$3
    local concurrency=$4
    local input_len=$5
    local output_len=$6
    local image_num=$7
    local resolution_name=$8
    
    local model_name=${model##*/}
    local cur_log_dir=${LOG_DIR}${model_name}_tp${tp_size}_pp${pp_size}
    mkdir -p ${cur_log_dir}

    local cur_case_name="${concurrency}_${image_num}_${resolution_name}"
    
    local addr="127.0.0.1"
    local port="8000"

    # 启动后端vllm服务
    echo "====================================="
    echo "正在测试 并发${concurrency} 图像数量${image_num} 图像分辨率${resolution_name} 场景"
    echo "-------------------------------------"
    echo "正在启动vllm推理服务： ${model} tp${tp_size}"
    local pid_list=()
    local cuda_graph_opts="--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"level\": 0}'"
    local CALCULATED_MAX_MODEL_LEN=$((input_len + output_len + 1000))
    # 修改多模态
    local server_cmd="VLLM_LOGGING_LEVEL=DEBUG NCCL_P2P_DISABLE=1  NCCL_IB_DISABLE=1 VLLM_ENFORCE_CUDA_GRAPH=1 vllm serve ${model} --trust-remote-code ${cuda_graph_opts}  --host ${addr} --port ${port} -tp ${tp_size} -pp ${pp_size} --gpu-memory-utilization 0.9  --max-model-len ${CALCULATED_MAX_MODEL_LEN} --limit-mm-per-prompt '{\"image\": 16, \"video\": 0}' &> ${cur_log_dir}/service_${cur_case_name}.log &" #  --mm-processor-kwargs max_pixels=1003520 
    echo $server_cmd
    eval $server_cmd
    pid_list+=($!)
    ping_server ${addr} ${port} ${pid_list[0]}
    if [[ $? != 0 ]]; then
        # 如果无法ping通，则退出本次测试
        return 1
    fi
    sleep 15
    # 增加获取kv cache size，并根据该size，判断当前并发数+语句长度是否会超出的判断。如果超出，则将特定变量设置为1，在外部执行对应处理。
    local num_gpu_block=$(awk '/vllm cache_config_info with initialization after num_gpu_blocks is/ {print $NF}' ${cur_log_dir}/service_${cur_case_name}.log)
    local num_max_tokens_supported=$((num_gpu_block * 16))
    local num_max_tokens_test=$((concurrency * (input_len + output_len + 1000)))
    if [[ $num_max_tokens_test -gt $num_max_tokens_supported ]]; then
        echo "当前case (输入 ${input_len} 输出 ${output_len} 并发 ${concurrency}) 会产生 $num_max_tokens_test 个token, 推理服务的num_gpu_blocks为 ${num_gpu_block} , 可容纳最多 ${num_max_tokens_supported} 个token, 不能满足测试需求, 跳过当前case"
        echo "-------------------------------------"
        cache_overflow=1
        # 清除后端服务
        echo "正在关闭vllm推理服务"
        cleanup
        echo "====================================="
        return 1
    fi
    echo "vllm推理服务启动完成"
    echo "-------------------------------------"

    # 启动client端测试
    run_client ${model} ${tp_size} ${pp_size} ${concurrency} ${input_len} ${output_len} ${image_num} ${resolution_name} ${cur_log_dir} 2>&1 |tee -a ${cur_log_dir}/client_${cur_case_name}.log

    # 清除后端服务
    echo "正在关闭vllm推理服务"
    cleanup
    echo "====================================="
}


### 设置测试case

# # 紫金山环境
target_model_cases=(
    "/tmp/gpumodel/Qwen3-VL-2B-Instruct/ 1"
)

concurrencies=(
    1
    2
    4
    8
    16
    32
    64
    128
    256
    512
)
input_lens=(
    128
    # 1024 
    # 8192 
    # 32648
)
output_lens=(
    128
    # 1024 
    # 8192 
    # 32648
)
image_nums=(
    1
    # 2
    # 4
    # 8
    # 16
)
# image_resolutions=(
#     '{(1280, 720, 1): 1.0}'
#     '{(1920, 1080, 1): 1.0}'
#     '{(3840, 2160, 1): 1.0}'
# )
image_resolutions=(
    720p
    # 1080p
    # 4k
)

cache_overflow=0  # 用于判断测试目标case是否会超出kv cache限制，导致请求排队抢占

### 开始测试
for target_case_str in "${target_model_cases[@]}"; do
    target_case=(${target_case_str})
    model_path=${target_case[0]}
    tp_size=${target_case[1]}
    if [[ ${#target_case[@]} -gt 2 ]]; then
        pp_size=${target_case[2]}
    else
        pp_size=1
    fi
    
    for input_len in "${input_lens[@]}"; do
        for output_len in "${output_lens[@]}"; do
            for concurrency in "${concurrencies[@]}"; do
                for image_num in "${image_nums[@]}"; do
                    for image_resolution in "${image_resolutions[@]}"; do
                        run ${model_path} ${tp_size} ${pp_size} ${concurrency} ${input_len} ${output_len} ${image_num} ${image_resolution}
                        if [[ $cache_overflow == 1 ]]; then
                            echo "已超出当前配置可容纳的token上限，跳过后续并发case"
                            cache_overflow=0  # 重置flag变量
                            break
                        fi
                    done
                done
            done
        done
    done
done

python3 get_data.py ${LOG_DIR}

    