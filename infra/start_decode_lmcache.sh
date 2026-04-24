#!/usr/bin/env bash
# =============================================================================
# start_decode_lmcache.sh — Run vLLM as KV consumer using LMCacheConnectorV1
# Run on: T4 VM (DECODE VM)
# Requires: Redis reachable at REDIS_HOST (running on prefill VM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${DECODE_PORT:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_DECODE:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL_DECODE:-0.85}"
REDIS_HOST="${REDIS_HOST:-${VM_PREFILL_IP:?'REDIS_HOST or VM_PREFILL_IP must be set'}}"
REDIS_PORT="${REDIS_PORT:-6379}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER_NAME="amlic-decode"

echo "============================================"
echo "  Starting DECODE container (LMCache KV consumer)"
echo "  Image:        vllm/vllm-openai:latest"
echo "  Model:        $MODEL"
echo "  Port:         $PORT"
echo "  Redis:        redis://${REDIS_HOST}:${REDIS_PORT}"
echo "  HF cache:     $HF_CACHE"
echo "============================================"

# Write LMCache config with resolved Redis host
mkdir -p /tmp/amlic-configs
cat > /tmp/amlic-configs/lmcache-decoder-config.yaml <<EOF
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_url: "redis://${REDIS_HOST}:${REDIS_PORT}"
remote_serde: "naive"
EOF

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false}}'

# Write startup script — single-quoted heredoc so nothing expands at write time;
# dynamic values are passed as -e env vars and expand inside the container.
cat > /tmp/amlic-start.sh <<'STARTEOF'
#!/bin/bash
set -e
pip install lmcache -q
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype float16 \
    --enforce-eager \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
STARTEOF
chmod +x /tmp/amlic-start.sh

sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo docker run -d --name "$CONTAINER_NAME" \
    --gpus all \
    --network host \
    --ipc host \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v /tmp/amlic-configs:/configs \
    -v /tmp/amlic-start.sh:/start.sh \
    -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL="$MODEL" \
    -e PORT="$PORT" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e KV_TRANSFER_CONFIG="$KV_TRANSFER_CONFIG" \
    -e LMCACHE_CONFIG_FILE="/configs/lmcache-decoder-config.yaml" \
    -e LMCACHE_USE_EXPERIMENTAL="True" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING="1" \
    -e VLLM_WORKER_MULTIPROC_METHOD="spawn" \
    -e PYTHONHASHSEED="42" \
    --entrypoint bash \
    vllm/vllm-openai:latest \
    /start.sh

echo ""
echo "Container started. Tail logs with:"
echo "  sudo docker logs -f $CONTAINER_NAME"
echo ""
echo "Health check (wait ~60s for model load):"
echo "  curl http://localhost:$PORT/health"
