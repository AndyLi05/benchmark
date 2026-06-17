#!/usr/bin/env python3
"""GSM8K benchmark for speculative decoding.

Usage:
    # Connect to draft API on 153:
    python benchmark.py --draft-url http://10.10.111.101:8866 --mode naive-greedy

    # All combos:
    for mode in naive-greedy naive-nongreedy pipesd-greedy pipesd-nongreedy; do
        python benchmark.py --draft-url http://10.10.111.101:8866 --mode $mode
    done
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests
from tqdm import tqdm


@dataclass
class Result:
    latency: float = 0.0       # total end-to-end latency (s)
    ttft: float = 0.0          # time to first token (s)
    tpot_list: list = field(default_factory=list)  # per-token latencies
    n_tokens: int = 0          # number of output tokens
    n_input_tokens: int = 0    # number of input tokens
    tpots: list = field(default_factory=list)
    error: str = ""


def load_gsm8k(split: str = "test", max_samples: int = 100) -> list[str]:
    """Load GSM8K questions from local parquet files."""
    import os

    local_paths = [
        "/root/share/datasets/gsm8k/test.parquet",
        "/home/share/datasets/gsm8k/test.parquet",
    ]
    parquet_path = None
    for p in local_paths:
        if os.path.exists(p):
            parquet_path = p
            break

    if parquet_path:
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            if "question" in df.columns:
                questions = df["question"].head(max_samples).tolist()
            elif "prompt" in df.columns:
                # Some datasets use 'prompt' column with nested dicts
                raw = df["prompt"].head(max_samples).tolist()
                questions = []
                for r in raw:
                    if isinstance(r, list) and len(r) > 0:
                        questions.append(r[0].get("content", str(r)))
                    elif isinstance(r, dict):
                        questions.append(r.get("content", str(r)))
                    else:
                        questions.append(str(r))
            else:
                raise KeyError(f"No known question column in {df.columns.tolist()}")
            print(f"Loaded {len(questions)} GSM8K questions from {parquet_path}")
            return questions
        except Exception as e:
            print(f"Parquet load failed: {e}")

    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=split)
        questions = [item["question"] for item in ds.select(range(min(max_samples, len(ds))))]
        print(f"Loaded {len(questions)} GSM8K {split} questions")
        return questions
    except Exception as e:
        print(f"Failed to load GSM8K: {e}")
        # Fallback questions
        return [
            "What is 2+2?",
            "If John has 5 apples and gives 2 to Mary, how many does he have left?",
            "A train travels at 60 mph for 2 hours. How far does it go?",
        ]


def send_request(
    draft_url: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stream: bool = True,
    timeout: int = 300,
) -> Result:
    """Send a request to the draft API and measure metrics."""
    result = Result()
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    try:
        t_start = time.monotonic()
        resp = requests.post(
            f"{draft_url}/v1/chat/completions",
            headers=headers,
            json=data,
            stream=stream,
            timeout=timeout,
        )
        resp.raise_for_status()

        if stream:
            tokens = []
            first_token_time = None
            prev_time = t_start
            tpot_times = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        tokens.append(content)
                        now = time.monotonic()
                        if first_token_time is None:
                            first_token_time = now
                            result.ttft = now - t_start
                        else:
                            tpot_times.append(now - prev_time)
                        prev_time = now
                except json.JSONDecodeError:
                    continue

            result.latency = time.monotonic() - t_start
            result.n_tokens = len(tokens)
            result.tpot_list = tpot_times
            if result.n_tokens > 1:
                result.tpots = tpot_times
            else:
                result.tpots = []
        else:
            t_start = time.monotonic()
            obj = resp.json()
            content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            result.latency = time.monotonic() - t_start
            result.n_tokens = len(content)
            result.ttft = result.latency  # non-streaming

    except Exception as e:
        result.error = str(e)

    return result


def run_benchmark(
    draft_url: str,
    questions: list[str],
    temperature: float,
    max_tokens: int,
    mode_name: str,
    num_workers: int = 4,
) -> dict:
    """Run benchmark over all questions."""
    results: list[Result] = []
    concurrency = min(num_workers, 8)

    print(f"\n{'='*60}")
    print(f"Benchmark: {mode_name}")
    print(f"  draft_url={draft_url}, temp={temperature}, max_tokens={max_tokens}")
    print(f"  questions={len(questions)}, concurrency={concurrency}")
    print(f"{'='*60}")

    with ThreadPoolExecutor(concurrency) as ex:
        futures = {
            ex.submit(send_request, draft_url, q, temperature, max_tokens): i
            for i, q in enumerate(questions)
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Requests"):
            results.append(f.result())

    # Compute stats
    valid = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    latencies = [r.latency for r in valid]
    ttfts = [r.ttft for r in valid if r.ttft > 0]
    all_tpots = []
    for r in valid:
        all_tpots.extend(r.tpot_list)
    tokens_per_req = [r.n_tokens for r in valid]
    tokens_total = sum(tokens_per_req)
    time_total = sum(latencies)

    stats = {
        "mode": mode_name,
        "temperature": temperature,
        "num_requests": len(questions),
        "num_valid": len(valid),
        "num_errors": len(errors),
        "latency_avg": sum(latencies) / len(latencies) if latencies else 0,
        "latency_p50": sorted(latencies)[len(latencies)//2] if latencies else 0,
        "latency_p95": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "latency_p99": sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
        "ttft_avg": sum(ttfts) / len(ttfts) if ttfts else 0,
        "ttft_p50": sorted(ttfts)[len(ttfts)//2] if ttfts else 0,
        "tpot_avg": sum(all_tpots) / len(all_tpots) if all_tpots else 0,
        "tpot_p50": sorted(all_tpots)[len(all_tpots)//2] if all_tpots else 0,
        "tpot_p99": sorted(all_tpots)[int(len(all_tpots)*0.99)] if all_tpots else 0,
        "tokens_per_req_avg": sum(tokens_per_req) / len(tokens_per_req) if tokens_per_req else 0,
        "edge_throughput": tokens_total / time_total if time_total > 0 else 0,
    }

    # Print results
    print(f"\n{'─'*60}")
    print(f"Results: {mode_name}")
    print(f"{'─'*60}")
    print(f"  Valid/Error:    {len(valid)}/{len(errors)}")
    print(f"  Latency avg:    {stats['latency_avg']:.2f}s")
    print(f"  Latency p50:    {stats['latency_p50']:.2f}s")
    print(f"  Latency p95:    {stats['latency_p95']:.2f}s")
    print(f"  TTFT avg:       {stats['ttft_avg']*1000:.1f}ms")
    print(f"  TPOT avg:       {stats['tpot_avg']*1000:.1f}ms")
    print(f"  TPOT p50:       {stats['tpot_p50']*1000:.1f}ms")
    print(f"  TPOT p99:       {stats['tpot_p99']*1000:.1f}ms")
    print(f"  Tokens/req:     {stats['tokens_per_req_avg']:.1f}")
    print(f"  Edge Throughput: {stats['edge_throughput']:.1f} tokens/s")
    print(f"{'─'*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-url", default="http://127.0.0.1:8866")
    parser.add_argument("--mode", choices=[
        "naive-greedy", "naive-nongreedy",
        "pipesd-greedy", "pipesd-nongreedy",
    ], required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    # Temperature mapping
    temp_map = {
        "naive-greedy": 0.0,
        "naive-nongreedy": 1.0,
        "pipesd-greedy": 0.0,
        "pipesd-nongreedy": 1.0,
    }

    questions = load_gsm8k("test", args.max_samples)
    stats = run_benchmark(
        draft_url=args.draft_url,
        questions=questions,
        temperature=temp_map[args.mode],
        max_tokens=args.max_tokens,
        mode_name=args.mode,
        num_workers=args.concurrency,
    )

    # Save results
    out_path = f"benchmark_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
