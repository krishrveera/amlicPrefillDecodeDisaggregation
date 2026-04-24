#!/usr/bin/env bash
# =============================================================================
# start_redis.sh — Run Redis in Docker on the PREFILL VM
# Redis serves as shared KV cache storage for LMCacheConnectorV1.
# Run on: PREFILL VM (L4)
# Usage:  bash infra/start_redis.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

CONTAINER_NAME="amlic-redis"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "============================================"
echo "  Starting Redis container"
echo "  Port:       0.0.0.0:${REDIS_PORT}"
echo "  Max memory: 4gb (allkeys-lru eviction)"
echo "  Container:  ${CONTAINER_NAME}"
echo "============================================"

sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo docker run -d --name "$CONTAINER_NAME" \
    --network host \
    --restart unless-stopped \
    redis:7-alpine \
    redis-server \
        --bind 0.0.0.0 \
        --port "$REDIS_PORT" \
        --maxmemory 4gb \
        --maxmemory-policy allkeys-lru \
        --save "" \
        --appendonly no

echo ""
echo "Redis started. Verify with:"
echo "  redis-cli -h 127.0.0.1 -p ${REDIS_PORT} ping"
echo "  sudo docker logs ${CONTAINER_NAME}"
