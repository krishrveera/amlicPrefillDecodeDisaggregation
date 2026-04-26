#!/usr/bin/env python3
"""
amlic_benchmark.py — latency measurement client for OpenAI-compatible chat completions.

Measures TTFT, E2E latency, ITL, and throughput across a sweep of prompt lengths.
Run separately per condition (collocated, lmcache, zmq); produces identical CSV output
for direct comparison.

Usage:
    python benchmark/amlic_benchmark.py \\
        --endpoint http://localhost:9000/v1/chat/completions \\
        --condition lmcache \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --runs 3 --max-tokens 100 --output benchmark/results/
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from tabulate import tabulate

PROMPTS = [
    {
        "label": "P1_tiny",
        "text": "Explain what a neural network is in simple terms.",
    },
    {
        "label": "P2_small",
        "text": (
            "Explain the difference between supervised and unsupervised learning in machine "
            "learning, giving two concrete examples of each type of approach and explaining "
            "what kinds of problems each is best suited to solve."
        ),
    },
    {
        "label": "P3_medium",
        "text": (
            "You are an expert machine learning engineer. Explain in detail how the transformer "
            "architecture works, covering the self-attention mechanism, multi-head attention, "
            "positional encoding, the encoder and decoder stacks, layer normalization, "
            "feed-forward networks, and how all these components combine to process sequential "
            "data. Include the mathematical intuition behind scaled dot-product attention."
        ),
    },
    {
        "label": "P4_large",
        "text": (
            "You are an expert in the history of computing, mathematics, cognitive science, "
            "neuroscience, linguistics, philosophy of mind, and artificial intelligence. Please "
            "provide an exhaustive and deeply technical account of the intellectual lineage of "
            "modern large language models, beginning with the foundational work of Alan Turing on "
            "computability and the Turing Test in 1950, proceeding through McCulloch and Pitts "
            "neural networks in 1943, Rosenblatt perceptrons in 1958, the first AI winter "
            "following the Minsky and Papert critique of perceptrons in 1969, the development of "
            "backpropagation by Rumelhart Hinton and Williams in 1986, the second AI winter of "
            "the late 1980s and early 1990s, the resurgence through support vector machines and "
            "kernel methods in the 1990s, the ImageNet moment and AlexNet in 2012, the "
            "development of word2vec and distributed word representations in 2013, the "
            "introduction of sequence to sequence models and attention mechanisms by Bahdanau Cho "
            "and Bengio in 2014 and 2015, the revolutionary Transformer architecture introduced "
            "in Attention Is All You Need by Vaswani et al in 2017, the pretraining and "
            "fine-tuning paradigm established by ELMo in 2018 and BERT and GPT in 2018 and 2019, "
            "the scaling laws discovered by Kaplan et al at OpenAI in 2020, the emergence of "
            "chain of thought reasoning in large models, the introduction of RLHF by Christiano "
            "et al and its application in InstructGPT and ChatGPT, and finally the current "
            "frontier of multimodal models reasoning models and the debate around whether scale "
            "alone is sufficient for artificial general intelligence."
        ),
    },
    {
        "label": "P5_xlarge",
        "text": (
            "You are a professor teaching a graduate seminar on distributed systems and machine "
            "learning infrastructure. Write a comprehensive lecture covering the following topics "
            "in depth: first, the fundamental problem of GPU memory bandwidth as a bottleneck in "
            "autoregressive language model decoding and why the memory wall problem gets worse as "
            "models scale; second, the concept of KV cache and why it grows linearly with "
            "sequence length and batch size; third, the motivation for prefill-decode "
            "disaggregation as proposed in DistServe and Splitwise and the theoretical "
            "performance benefits; fourth, the engineering challenges of KV cache transfer "
            "between nodes including serialization overhead bandwidth requirements and network "
            "latency constraints; fifth, the design tradeoffs between homogeneous and "
            "heterogeneous GPU configurations for disaggregated serving and how attention backend "
            "compatibility affects feasibility; sixth, current production implementations at "
            "companies like Meta LinkedIn and Mistral using vLLM with disaggregated serving; "
            "seventh, the role of NIXL UCX and emerging KV transfer libraries like LMCache in "
            "making disaggregation practical; and eighth, open research questions around dynamic "
            "routing threshold computation the crossover point N where disaggregation becomes "
            "beneficial versus collocated serving and how workload characteristics affect this "
            "threshold."
        ),
    },
]


async def measure_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    ttft_ms = None
    e2e_ms = None
    prompt_tokens = None
    output_tokens = None
    status = "success"
    t_start = time.perf_counter()

    try:
        async with client.stream(
            "POST", endpoint, json=payload, timeout=timeout
        ) as response:
            if response.status_code != 200:
                return {
                    "ttft_ms": None,
                    "e2e_ms": None,
                    "itl_ms": None,
                    "throughput_tps": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "status": f"failed_{response.status_code}",
                }

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # capture TTFT on first content token
                if ttft_ms is None:
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            ttft_ms = (time.perf_counter() - t_start) * 1000

                # capture usage from any chunk that has it
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")

        e2e_ms = (time.perf_counter() - t_start) * 1000

    except httpx.TimeoutException as exc:
        return {
            "ttft_ms": None,
            "e2e_ms": None,
            "itl_ms": None,
            "throughput_tps": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "status": f"failed_timeout: {exc}",
        }
    except httpx.RequestError as exc:
        return {
            "ttft_ms": None,
            "e2e_ms": None,
            "itl_ms": None,
            "throughput_tps": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "status": f"failed_connection: {exc}",
        }

    itl_ms = None
    if (
        ttft_ms is not None
        and e2e_ms is not None
        and output_tokens is not None
        and output_tokens > 1
    ):
        itl_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)

    throughput_tps = None
    if output_tokens is not None and e2e_ms is not None and e2e_ms > 0:
        throughput_tps = output_tokens / (e2e_ms / 1000)

    return {
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "e2e_ms": round(e2e_ms, 2) if e2e_ms is not None else None,
        "itl_ms": round(itl_ms, 2) if itl_ms is not None else None,
        "throughput_tps": round(throughput_tps, 3) if throughput_tps is not None else None,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "status": status,
    }


async def run_condition(
    endpoint: str,
    condition: str,
    model: str,
    prompts: list[dict],
    runs: int,
    max_tokens: int,
    warmup: int,
    timeout: float,
) -> list[dict]:
    results = []

    async with httpx.AsyncClient() as client:
        # warmup
        for prompt in prompts:
            for w in range(warmup):
                print(f"[Warmup] {prompt['label']}...", end=" ", flush=True)
                t0 = time.perf_counter()
                r = await measure_request(
                    client, endpoint, model, prompt["text"], max_tokens, timeout
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                print(f"done ({elapsed}ms)")

        print()

        # timed runs
        for idx, prompt in enumerate(prompts):
            label = prompt["label"]
            print(f"[{idx + 1}/{len(prompts)}] {label}")
            run_results = []

            for run_num in range(1, runs + 1):
                r = await measure_request(
                    client, endpoint, model, prompt["text"], max_tokens, timeout
                )
                run_results.append(r)

                ttft_disp = f"{r['ttft_ms']:.0f}ms" if r["ttft_ms"] is not None else "N/A"
                e2e_disp  = f"{r['e2e_ms']:.0f}ms"  if r["e2e_ms"]  is not None else "N/A"
                itl_disp  = f"{r['itl_ms']:.0f}ms"  if r["itl_ms"]  is not None else "N/A"
                tps_disp  = f"{r['throughput_tps']:.1f}" if r["throughput_tps"] is not None else "N/A"
                print(
                    f"  Run {run_num}: TTFT={ttft_disp}  E2E={e2e_disp}"
                    f"  ITL={itl_disp}  TPS={tps_disp}"
                )

                results.append({
                    "condition":       condition,
                    "prompt_label":    label,
                    "prompt_tokens":   r["prompt_tokens"],
                    "run":             run_num,
                    "ttft_ms":         r["ttft_ms"],
                    "e2e_ms":          r["e2e_ms"],
                    "itl_ms":          r["itl_ms"],
                    "throughput_tps":  r["throughput_tps"],
                    "output_tokens":   r["output_tokens"],
                    "status":          r["status"],
                })

            # per-prompt mean line
            good = [x for x in run_results if x["status"] == "success"]
            if good:
                def _mean(key):
                    vals = [x[key] for x in good if x[key] is not None]
                    return statistics.mean(vals) if vals else None

                m_ttft = _mean("ttft_ms")
                m_e2e  = _mean("e2e_ms")
                m_itl  = _mean("itl_ms")
                m_tps  = _mean("throughput_tps")
                print(
                    f"  Mean:  TTFT={m_ttft:.0f}ms  E2E={m_e2e:.0f}ms"
                    f"  ITL={m_itl:.0f}ms  TPS={m_tps:.1f}"
                    if all(v is not None for v in [m_ttft, m_e2e, m_itl, m_tps])
                    else "  Mean:  (some metrics unavailable)"
                )
            else:
                print(f"  All {runs} runs failed for {label}.")
            print()

    return results


def _stats(values: list) -> dict:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(clean), 2),
        "std":  round(statistics.stdev(clean), 2) if len(clean) > 1 else 0.0,
        "min":  round(min(clean), 2),
        "max":  round(max(clean), 2),
    }


def write_results(
    results: list[dict],
    condition: str,
    endpoint: str,
    runs: int,
    max_tokens: int,
    warmup: int,
    output_dir: str,
) -> tuple[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = str(output_path / f"{condition}_{ts}.csv")
    json_path = str(output_path / f"{condition}_{ts}_summary.json")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    per_prompt = {}
    for label, grp in df.groupby("prompt_label"):
        good = grp[grp["status"] == "success"]
        per_prompt[label] = {
            "prompt_tokens":  int(grp["prompt_tokens"].dropna().iloc[0]) if not grp["prompt_tokens"].dropna().empty else None,
            "ttft_ms":        _stats(good["ttft_ms"].tolist()),
            "e2e_ms":         _stats(good["e2e_ms"].tolist()),
            "itl_ms":         _stats(good["itl_ms"].tolist()),
            "throughput_tps": _stats(good["throughput_tps"].tolist()),
            "success_rate":   round(len(good) / len(grp), 3),
        }

    summary = {
        "condition": condition,
        "timestamp": ts,
        "endpoint":  endpoint,
        "config": {
            "runs":       runs,
            "max_tokens": max_tokens,
            "warmup":     warmup,
        },
        "results": per_prompt,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def print_summary_table(summary_json_path: str):
    with open(summary_json_path) as f:
        summary = json.load(f)

    rows = []
    for label in sorted(summary["results"]):
        r = summary["results"][label]
        rows.append([
            label,
            r["prompt_tokens"],
            f"{r['ttft_ms']['mean']:.0f}"        if r["ttft_ms"]["mean"]        is not None else "N/A",
            f"{r['e2e_ms']['mean']:.0f}"          if r["e2e_ms"]["mean"]         is not None else "N/A",
            f"{r['itl_ms']['mean']:.0f}"          if r["itl_ms"]["mean"]         is not None else "N/A",
            f"{r['throughput_tps']['mean']:.1f}"  if r["throughput_tps"]["mean"] is not None else "N/A",
            f"{r['success_rate']:.0%}",
        ])

    print(tabulate(
        rows,
        headers=["prompt_label", "prompt_tokens", "ttft_mean_ms", "e2e_mean_ms",
                 "itl_mean_ms", "tps_mean", "success_rate"],
        tablefmt="simple",
    ))


def main():
    ap = argparse.ArgumentParser(description="AMLIC latency benchmark")
    ap.add_argument("--endpoint",   required=True,  help="Full URL of chat completions endpoint")
    ap.add_argument("--condition",  required=True,  help="Label for this run (collocated/lmcache/zmq)")
    ap.add_argument("--model",      default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--runs",       type=int,   default=3)
    ap.add_argument("--max-tokens", type=int,   default=100)
    ap.add_argument("--warmup",     type=int,   default=1)
    ap.add_argument("--output",     default="benchmark/results/")
    ap.add_argument("--timeout",    type=float, default=120.0)
    args = ap.parse_args()

    print(f"\nAMLIC Benchmark — condition: {args.condition}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs} | Max tokens: {args.max_tokens} | Warmup: {args.warmup}\n")

    results = asyncio.run(run_condition(
        endpoint=args.endpoint,
        condition=args.condition,
        model=args.model,
        prompts=PROMPTS,
        runs=args.runs,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        timeout=args.timeout,
    ))

    # per-prompt success summary
    df = pd.DataFrame(results)
    for label, grp in df.groupby("prompt_label"):
        n_ok   = (grp["status"] == "success").sum()
        n_fail = len(grp) - n_ok
        if n_fail:
            print(f"  {label}: {n_ok}/{len(grp)} runs succeeded ({n_fail} failed)")

    csv_path, json_path = write_results(
        results=results,
        condition=args.condition,
        endpoint=args.endpoint,
        runs=args.runs,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        output_dir=args.output,
    )

    print(f"\nResults written to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(f"\nSummary Table:")
    print_summary_table(json_path)
    print()


if __name__ == "__main__":
    main()
