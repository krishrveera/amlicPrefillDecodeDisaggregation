# Prefill-Decode Disaggregated Inference Serving

> Prefill/Decode Disaggregation for LLM Serving using vLLM + LMCache

## Overview

This project implements and benchmarks **collocated** vs **disaggregated** (prefill/decode) LLM serving. In disaggregated mode, a dedicated GPU handles prompt prefill while a separate GPU handles autoregressive decode, with KV cache transferred between them via [LMCache](https://github.com/LMCache/LMCache) through Redis.

### Architecture

```
┌─────────────────┐         ┌──────────────────────────────────┐
│   User / Demo   │         │         Adaptive Router          │
│   (Streamlit)   │────────▶│  token_count ≥ N? → disagg      │
│                 │         │  token_count < N? → collocated   │
└─────────────────┘         └──────────┬───────────┬───────────┘
                                       │           │
                            ┌──────────▼──┐   ┌────▼───────────────────────┐
                            │ Collocated  │   │   Disaggregated Proxy     │
                            │  vLLM (L4)  │   │   (FastAPI)               │
                            │  Port 8000  │   │   Port 9000               │
                            └─────────────┘   └────┬──────────┬───────────┘
                                                   │          │
                                          ┌────────▼──┐  ┌────▼────────┐
                                          │  Prefill  │  │   Decode    │
                                          │ vLLM (L4) │  │  vLLM (L4) │
                                          │ Port 8100 │  │  Port 8200  │
                                          │ KV Prod.  │  │  KV Cons.  │
                                          └─────┬─────┘  └─────┬──────┘
                                                │              │
                                          ┌─────▼──────────────▼──────┐
                                          │     Redis (LMCache)       │
                                          │  Shared KV Cache Storage  │
                                          └───────────────────────────┘
```

### Hardware Requirements

| Role                 | GPU                       | VRAM     | Example VM            |
| -------------------- | ------------------------- | -------- | --------------------- |
| Prefill + Collocated | NVIDIA L4 (or equivalent) | ≥ 24 GB | GCP `g2-standard-4` |
| Decode               | NVIDIA L4 (or equivalent) | ≥ 24 GB | GCP `g2-standard-4` |

> **Note:** Both GPUs must use the same attention backend and dtype for LMCache KV cache compatibility. We used two L4 GPUs. A T4 decode GPU works but requires matching attention configs.

### Model

- `meta-llama/Llama-3.2-3B-Instruct` (3B params, fits in 16 GB with room for KV cache)
- Requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to Meta Llama models

---

## Repository Structure

```
├── infra/                        # Infrastructure & server scripts
│   ├── vm_bootstrap.sh           # Idempotent VM setup (drivers, venv, vLLM, LMCache)
│   ├── start_collocated.sh       # Launch collocated vLLM (single GPU)
│   ├── start_prefill_lmcache.sh  # Launch prefill vLLM (KV producer, Docker)
│   ├── start_decode_lmcache.sh   # Launch decode vLLM (KV consumer, Docker)
│   ├── start_redis.sh            # Launch Redis for KV cache sharing
│   ├── proxy_server.py           # FastAPI proxy: prefill → Redis → decode
│   └── configs/                  # LMCache YAML configs
│
├── benchmark/                    # Benchmarking & analysis
│   ├── amlic_benchmark.py        # Prompt-sweep latency benchmark
│   ├── poisson_load.py           # Poisson-arrival load generator
│   ├── client.py                 # Async OpenAI-compatible streaming client
│   ├── workloads.py              # Workload profiles + prompt generation
│   ├── compute_threshold.py      # Compute crossover threshold N
│   ├── plot_results.py           # Generate benchmark plots
│   └── figures_vllm_final/       # Final benchmark result plots
│
├── router/                       # Adaptive router
│   └── router.py                 # Routes by prompt length threshold
│
├── demo/                         # Interactive demo
│   └── app.py                    # Streamlit chatbot UI
│
├── scripts/                      # Helper scripts
│   ├── run_full_benchmark.sh     # Run complete benchmark suite
│   ├── health_check.sh           # Check all services are healthy
│   └── start_amlic.py            # Orchestrated startup
```

---

## Replication Guide

### Prerequisites

- **Two GCP VMs** (or equivalent) with NVIDIA GPUs (L4 recommended)
- **Docker** installed on both VMs
- **Tailscale** (or any overlay network / direct connectivity between VMs)
- **HuggingFace account** with access to `meta-llama/Llama-3.2-3B-Instruct`
- **Python 3.10+** on your local machine (for running benchmarks)

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/amlicPrefillDecodeDisaggregation.git
cd amlicPrefillDecodeDisaggregation
```

### Step 2: Configure Environment

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required: Your HuggingFace token
HF_TOKEN=hf_your_token_here

# VM IPs (fill after Step 3)
VM_PREFILL_IP=<PREFILL_VM_IP>
VM_DECODE_IP=<DECODE_VM_IP>

# Ports (defaults work fine)
COLLOCATED_PORT=8000
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=9000
REDIS_PORT=6379
```

### Step 3: Provision GCP VMs

Create two GPU VMs. Example using `gcloud`:

```bash
# Prefill VM (L4 GPU)
gcloud compute instances create prefill-vm \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE

# Decode VM (L4 GPU)
gcloud compute instances create decode-vm \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

### Step 4: Install Tailscale on Both VMs

```bash
# SSH into each VM and run:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Note the Tailscale IP and update .env
tailscale ip -4
```

Update `VM_PREFILL_IP` and `VM_DECODE_IP` in your `.env` with the Tailscale IPs.

### Step 5: Bootstrap Both VMs

Copy the repo to each VM and run the bootstrap script:

```bash
# From your local machine:
scp -r . user@<PREFILL_VM_IP>:~/amlic/
scp -r . user@<DECODE_VM_IP>:~/amlic/

# SSH into each VM and run:
cd ~/amlic
bash infra/vm_bootstrap.sh
```

This installs: system packages, Python venv, vLLM, LMCache, and verifies the GPU driver.

### Step 6: Start the Collocated Baseline

On the **Prefill VM**:

```bash
cd ~/amlic
bash infra/start_collocated.sh
```

Wait ~60s for the model to load, then verify:

```bash
curl http://localhost:8000/health
```

### Step 7: Start the Disaggregated System

This requires 4 components started in order:

#### 7a. Start Redis (Prefill VM)

```bash
bash infra/start_redis.sh
```

Verify: `redis-cli -h 127.0.0.1 ping` → `PONG`

#### 7b. Start Prefill vLLM (Prefill VM)

```bash
bash infra/start_prefill_lmcache.sh
```

Wait ~60s, verify: `curl http://localhost:8100/health`

#### 7c. Start Decode vLLM (Decode VM)

```bash
bash infra/start_decode_lmcache.sh
```

Wait ~60s, verify: `curl http://localhost:8200/health`

#### 7d. Start Proxy (Prefill VM)

```bash
source ~/amlic-venv/bin/activate
python infra/proxy_server.py \
    --prefiller-host $(tailscale ip -4) --prefiller-port 8100 \
    --decoder-host <DECODE_TAILSCALE_IP> --decoder-port 8200 \
    --port 9000
```

Verify: `curl http://localhost:9000/health`

### Step 8: Run Benchmarks

From your **local machine** (or any machine that can reach the VMs):

```bash
pip install -e ".[dev]"
```

#### Prompt-Length Sweep

```bash
# Collocated
python benchmark/amlic_benchmark.py \
    --endpoint http://<PREFILL_IP>:8000/v1/chat/completions \
    --condition collocated \
    --runs 3 --max-tokens 100

# Disaggregated
python benchmark/amlic_benchmark.py \
    --endpoint http://<PREFILL_IP>:9000/v1/chat/completions \
    --condition disaggregated \
    --runs 3 --max-tokens 100
```

#### Poisson Load Test

```bash
# Collocated (4 req/s for 60s)
python -m benchmark.poisson_load \
    --endpoint http://<PREFILL_IP>:8000/v1 \
    --arch collocated \
    --rate 4.0 --duration 60

# Disaggregated
python -m benchmark.poisson_load \
    --endpoint http://<PREFILL_IP>:9000/v1 \
    --arch disaggregated \
    --rate 4.0 --duration 60
```

Results are saved as CSV files in `benchmark/results/`.

### Step 9: Launch the Demo (Optional)

```bash
pip install streamlit
streamlit run demo/app.py
```

---

## Key Metrics

| Metric                | Description                                               |
| --------------------- | --------------------------------------------------------- |
| **TTFT**        | Time to first token (ms) — user-perceived responsiveness |
| **TPOT**        | Time per output token (ms) — decode speed                |
| **Throughput**  | Output tokens per second                                  |
| **E2E Latency** | Total request latency from submission to last token       |

## How It Works

1. **Collocated**: A single GPU runs both prefill (processing the prompt) and decode (generating tokens). Under concurrent load, prefill and decode contend for GPU compute cycles.
2. **Disaggregated**: GPU 1 runs prefill only, GPU 2 runs decode only. After prefill, the KV cache is serialized and sent through Redis (via LMCache) to the decode GPU. This eliminates prefill-decode contention but adds KV transfer overhead.
3. **Proxy Server**: A FastAPI proxy (`infra/proxy_server.py`) coordinates the flow: it sends the prompt to prefill with `max_tokens=1` (triggering KV cache production), then forwards the full request to decode for token generation.
4. **Adaptive Router**: Routes requests to collocated or disaggregated based on prompt length threshold N.

## Troubleshooting

| Issue                    | Solution                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `nvidia-smi` not found | Install NVIDIA drivers or use a GCP Deep Learning VM image                           |
| Model download fails     | Verify `HF_TOKEN` has gated access to Llama models                                 |
| Redis connection refused | Check `REDIS_HOST` in `.env` matches Tailscale IP, ensure port 6379 is reachable |
| KV cache transfer fails  | Both GPUs must use same dtype (`float16`) and attention backend                    |
| Decode returns empty     | Check Redis has data:`redis-cli -h <REDIS_HOST> DBSIZE`                            |
| Proxy timeout            | Increase `httpx` timeout; verify both prefill and decode endpoints are healthy     |
