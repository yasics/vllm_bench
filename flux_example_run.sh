#!/bin/bash

# 1. 后台启动 vLLM 服务，并记录进程ID (PID)
nohup env VLLM_ENFORCE_CUDA_GRAPH=1 vllm serve /nvme/hztest/model/FLUX.2-klein-4B --omni --port 8092 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 > vllm.log 2>&1 &
VLLM_PID=$! # 获取后台进程的 PID 并保存到变量

# 2. 等待 vLLM 服务完全启动
echo "Waiting for vLLM service to be ready..."
while ! (echo > /dev/tcp/127.0.0.1/8092) 2>/dev/null; do
  sleep 1
done
echo "vLLM service is ready!"

# 3. 顺序执行三个 Python 测试指令
python benchtest.py   --url http://127.0.0.1:8092/v1/chat/completions   --model black-forest-labs/FLUX.2-klein-4B   --input-images ./input.png   --prompt "Convert this image to watercolor style"   --height 720   --width 1280   --steps 30   --guidance-scale 1   --mode concurrency   --concurrency 1   --num-requests 10   --warmup-requests 3   --vary-seed  --save-dir output --save-prob 1.0 --output-json report_conc1.json

python benchtest.py   --url http://127.0.0.1:8092/v1/chat/completions   --model black-forest-labs/FLUX.2-klein-4B   --input-images ./input.png   --prompt "Convert this image to watercolor style"   --height 720   --width 1280   --steps 30   --guidance-scale 1   --mode concurrency   --concurrency 2   --num-requests 20   --vary-seed  --save-dir output --save-prob 1.0 --output-json report_conc2.json

python benchtest.py   --url http://127.0.0.1:8092/v1/chat/completions   --model black-forest-labs/FLUX.2-klein-4B   --input-images ./input.png   --prompt "Convert this image to watercolor style"   --height 720   --width 1280   --steps 30   --guidance-scale 1   --mode concurrency   --concurrency 4   --num-requests 40   --vary-seed  --save-dir output --save-prob 1.0 --output-json report_conc4.json

# 4. 所有测试完成后，优雅地终止 vLLM 服务
echo "All tests completed. Shutting down vLLM service..."

# 4.1 发送 SIGTERM 信号，请求进程优雅退出
kill -15 $VLLM_PID
echo "Sent SIGTERM to vLLM (PID: $VLLM_PID), waiting for it to exit..."

# 4.2 等待进程退出，最多等待 30 秒
for i in {1..30}; do
    # 检查进程是否还存在
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "vLLM service has been stopped."
        break
    fi
    sleep 1
done

# 4.3 如果进程仍在运行，则强制结束
if ps -p $VLLM_PID > /dev/null; then
    echo "vLLM did not stop gracefully. Forcing shutdown with SIGKILL..."
    kill -9 $VLLM_PID
    echo "vLLM service forcefully stopped."
fi
