#!/usr/bin/env bash
# =============================================================================
# test_disagg_lmcache.sh — End-to-end smoke test for LMCache disagg pipeline
# Run on: LAPTOP (with .env populated and proxy reachable)
# Usage:  bash scripts/test_disagg_lmcache.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

REDIS_HOST="${REDIS_HOST:-${VM_PREFILL_IP:?'Set REDIS_HOST or VM_PREFILL_IP in .env'}}"
REDIS_PORT="${REDIS_PORT:-6379}"
VM_PREFILL_IP="${VM_PREFILL_IP:?'Set VM_PREFILL_IP in .env'}"
VM_DECODE_IP="${VM_DECODE_IP:?'Set VM_DECODE_IP in .env'}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-9000}"
MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" = "ok" ]; then
        echo "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label — $result"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "============================================"
echo "  AMLIC LMCache Disagg Smoke Test"
echo "  $(date)"
echo "============================================"
echo ""

# ── 1. Redis reachability ────────────────────────────────────────────────────
echo "[1/6] Checking Redis..."
REDIS_PING=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>&1 || echo "FAIL")
if [ "$REDIS_PING" = "PONG" ]; then
    check "Redis ping ${REDIS_HOST}:${REDIS_PORT}" "ok"
else
    check "Redis ping ${REDIS_HOST}:${REDIS_PORT}" "$REDIS_PING"
fi

# ── 2. Prefill vLLM health ───────────────────────────────────────────────────
echo "[2/6] Checking prefill vLLM (${VM_PREFILL_IP}:${PREFILL_PORT})..."
PREFILL_HEALTH=$(curl -sf --max-time 5 "http://${VM_PREFILL_IP}:${PREFILL_PORT}/health" 2>&1 || echo "FAIL")
if echo "$PREFILL_HEALTH" | grep -q "{}"; then
    check "Prefill vLLM health" "ok"
else
    check "Prefill vLLM health" "$PREFILL_HEALTH"
fi

# ── 3. Decode vLLM health ────────────────────────────────────────────────────
echo "[3/6] Checking decode vLLM (${VM_DECODE_IP}:${DECODE_PORT})..."
DECODE_HEALTH=$(curl -sf --max-time 5 "http://${VM_DECODE_IP}:${DECODE_PORT}/health" 2>&1 || echo "FAIL")
if echo "$DECODE_HEALTH" | grep -q "{}"; then
    check "Decode vLLM health" "ok"
else
    check "Decode vLLM health" "$DECODE_HEALTH"
fi

# ── 4. Start proxy ───────────────────────────────────────────────────────────
echo "[4/6] Starting proxy server on port ${PROXY_PORT}..."
# Kill any existing proxy on that port
lsof -ti ":${PROXY_PORT}" | xargs kill -9 2>/dev/null || true
sleep 1

python infra/proxy_server.py \
    --prefiller-host "$VM_PREFILL_IP" \
    --prefiller-port "$PREFILL_PORT" \
    --decoder-host "$VM_DECODE_IP" \
    --decoder-port "$DECODE_PORT" \
    --port "$PROXY_PORT" &
PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"
sleep 3

PROXY_HEALTH=$(curl -sf --max-time 5 "http://localhost:${PROXY_PORT}/health" 2>&1 || echo "FAIL")
if echo "$PROXY_HEALTH" | grep -q "ok"; then
    check "Proxy health" "ok"
else
    check "Proxy health" "$PROXY_HEALTH"
fi

# ── 5. End-to-end inference through proxy ───────────────────────────────────
echo "[5/6] Sending test request through proxy..."
RESPONSE=$(curl -s --max-time 120 "http://localhost:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}],
        \"max_tokens\": 30
    }" 2>&1 || echo "FAIL")

if echo "$RESPONSE" | grep -q '"content"'; then
    check "End-to-end inference" "ok"
    GENERATED=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "(parse error)")
    echo "  Response: $GENERATED"
else
    check "End-to-end inference" "$RESPONSE"
fi

# ── 6. Verify Redis has KV cache keys ───────────────────────────────────────
echo "[6/6] Checking Redis for KV cache keys..."
sleep 2
REDIS_KEYS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" DBSIZE 2>&1 || echo "FAIL")
if [ "$REDIS_KEYS" != "FAIL" ] && [ "$REDIS_KEYS" -gt "0" ] 2>/dev/null; then
    check "Redis has KV cache keys (count: ${REDIS_KEYS})" "ok"
    echo "  Sample keys:"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" KEYS '*' 2>/dev/null | head -5 | sed 's/^/    /'
else
    check "Redis KV cache keys" "0 keys found (KV may not have been stored)"
fi

# ── Cleanup ──────────────────────────────────────────────────────────────────
kill "$PROXY_PID" 2>/dev/null || true

echo ""
echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
