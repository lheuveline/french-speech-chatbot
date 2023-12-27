#!/bin/bash

python3 -m vllm.entrypoints.api_server \
  --model=${MODEL} \
  --quantization ${QUANTIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
