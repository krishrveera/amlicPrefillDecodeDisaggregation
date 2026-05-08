"""
poisson_load.py — Poisson-arrival benchmark client.

Generates a request stream with exponentially-distributed inter-arrivals and
mixed prompt sizes, fires them at scheduled times, and records per-request
timings + the number of requests in-flight when each one started.

Usage:
    python -m benchmark.poisson_load \
        --endpoint http://localhost:9000/v1 \
        --arch disaggregated \
        --rate 4.0 \
        --duration 60 \
        --output-tokens 200 \
        --seed 42
"""

import argparse
import asyncio
import csv
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.client import send_streaming_request
from benchmark.workloads import load_source_text, make_prompt

PROMPT_SIZES = [100, 500, 1000, 1500]


async def run_poisson(endpoint, arch, rate, duration, seed, output_tokens, out_dir, model):
    random.seed(seed)
    source = load_source_text()

    schedule = []
    t = 0.0
    while True:
        t += random.expovariate(rate)
        if t >= duration:
            break
        schedule.append((t, random.choice(PROMPT_SIZES)))

    dist = {p: sum(1 for _, sz in schedule if sz == p) for p in PROMPT_SIZES}
    print(f"Generated {len(schedule)} requests over {duration}s at lambda={rate} req/s")
    print(f"Prompt size mix: {dist}")

    requests = []
    for i, (arrival, pz) in enumerate(schedule):
        prompt, ptoks = make_prompt(pz, source, model)
        requests.append({"id": i, "arrival": arrival, "prompt": prompt,
                         "ptoks": ptoks, "target_pz": pz})

    inflight = 0
    inflight_max = 0
    arrival_inflight = {}
    inflight_lock = asyncio.Lock()

    async def fire(session, req, T0):
        nonlocal inflight, inflight_max
        delay = req["arrival"] - (time.perf_counter() - T0)
        if delay > 0:
            await asyncio.sleep(delay)
        async with inflight_lock:
            inflight += 1
            inflight_max = max(inflight_max, inflight)
            arrival_inflight[req["id"]] = inflight
        result = await send_streaming_request(
            session, endpoint, req["prompt"], output_tokens,
            req["id"], req["ptoks"], model,
        )
        async with inflight_lock:
            inflight -= 1
        return req, result

    print(f"\nFiring against {endpoint} ...")
    timeout = aiohttp.ClientTimeout(total=600)
    T_wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        T0 = time.perf_counter()
        tasks = [fire(session, r, T0) for r in requests]
        all_results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - T_wall_start

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"{arch}_poisson_lambda{rate:.0f}_{ts}.csv"

    with fname.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arch", "lambda", "request_id", "arrival_time_ms",
            "in_flight_at_start", "target_prompt_tokens", "prompt_tokens",
            "output_tokens", "ttft_ms", "itl_ms_mean", "itl_ms_p50",
            "itl_ms_p95", "throughput_tps", "total_latency_ms", "error",
        ])
        for req, r in all_results:
            w.writerow([
                arch, rate, r.request_id, round(req["arrival"] * 1000, 2),
                arrival_inflight.get(req["id"], -1),
                req["target_pz"], r.prompt_tokens, r.output_tokens,
                round(r.ttft_ms, 2), round(r.itl_ms_mean, 2),
                round(r.itl_ms_p50, 2), round(r.itl_ms_p95, 2),
                round(r.throughput_tps, 2), round(r.total_latency_ms, 2),
                r.error or "",
            ])

    ok = [r for _, r in all_results if r.error is None]
    bad = [r for _, r in all_results if r.error is not None]
    if ok:
        ttfts = sorted(r.ttft_ms for r in ok)
        totals = sorted(r.total_latency_ms for r in ok)
        itls = [r.itl_ms_mean for r in ok]
        agg = sum(r.output_tokens for r in ok) / wall_time
        def pct(xs, p): return xs[min(int(len(xs) * p), len(xs) - 1)]
        print(f"\nResults  (wall={wall_time:.1f}s, max_in_flight={inflight_max})")
        print(f"  successful: {len(ok)} / {len(all_results)}  (failed={len(bad)})")
        print(f"  TTFT  mean / p50 / p95: {statistics.mean(ttfts):.0f} / {pct(ttfts,0.5):.0f} / {pct(ttfts,0.95):.0f}  ms")
        print(f"  Total mean / p50 / p95: {statistics.mean(totals):.0f} / {pct(totals,0.5):.0f} / {pct(totals,0.95):.0f}  ms")
        print(f"  ITL   mean:             {statistics.mean(itls):.1f} ms")
        print(f"  Aggregate throughput:   {agg:.1f} tokens/s")
    print(f"  saved: {fname}")
    return fname


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--arch", required=True, choices=["collocated", "disaggregated"])
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-tokens", type=int, default=200)
    p.add_argument("--out", default="benchmark/results")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    a = p.parse_args()
    asyncio.run(run_poisson(a.endpoint, a.arch, a.rate, a.duration,
                            a.seed, a.output_tokens, a.out, a.model))


if __name__ == "__main__":
    main()
