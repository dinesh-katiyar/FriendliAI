"""CLI benchmark: Friendli vs vLLM streaming chat (throughput, TTFT, ITL)."""
import asyncio, time, json, os, argparse, statistics
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import aiohttp
    import matplotlib.pyplot as plt
except ImportError:
    print("Install: pip install aiohttp matplotlib")
    exit(1)

@dataclass
class RequestResult:
    """Per-request metrics from one streaming completion (tokens as deltas)."""

    ttft: float = 0.0
    itl_values: list = field(default_factory=list)
    total_tokens: int = 0
    total_time: float = 0.0
    error: Optional[str] = None

PROMPTS = [
    "Explain how KV caching works in transformer inference.",
    "What are the trade-offs between tensor and pipeline parallelism?",
    "Describe the ORCA scheduling algorithm for LLM inference.",
    "Compare continuous batching with static batching.",
    "Explain PagedAttention and its memory management benefits.",
    "Write a Kubernetes StatefulSet YAML for GPU inference.",
    "What is speculative decoding and how does it reduce latency?",
]

@dataclass
class EngineConfig:
    """Target engine: display name, OpenAI-style base URL (/v1), model id, optional API key."""

    name: str
    base_url: str
    model: str
    api_key: str = ""

    @property
    def headers(self) -> dict:
        """JSON Content-Type and Bearer Authorization when ``api_key`` is non-empty."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

async def send_streaming_request(session, config: EngineConfig, prompt: str) -> RequestResult:
    """POST ``{base_url}/chat/completions`` with ``stream=True`` and record timings.

    Parses SSE ``data:`` lines; counts content deltas as tokens. On failure, sets ``result.error``.
    """
    result = RequestResult()
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
    }
    # FriendliAI dedicated endpoint: POST {base_url}/chat/completions
    # base_url already ends with /v1, OpenAI SDK appends /chat/completions
    url = f"{config.base_url}/chat/completions"
    start = time.perf_counter()
    first_token_time = None
    last_token_time = start
    try:
        async with session.post(
            url, json=payload, headers=config.headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        result.ttft = now - start
                    else:
                        result.itl_values.append(now - last_token_time)
                    last_token_time = now
                    result.total_tokens += 1
        result.total_time = time.perf_counter() - start
    except Exception as e:
        result.error = str(e) 
    return result

async def run_concurrency_level(
    config: EngineConfig, concurrency: int, num_requests: int
) -> Optional[dict[str, Any]]:
    """Run ``num_requests`` streaming calls capped at ``concurrency`` in flight.

    Returns a metrics dict (throughput, avg TTFT/ITL, success rate) or ``None`` if all failed.
    """
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def limited_request(prompt):
            async with semaphore:
                return await send_streaming_request(session, config, prompt)
        tasks = [limited_request(PROMPTS[i % len(PROMPTS)]) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    valid = [r for r in results if r.error is None and r.total_tokens > 0]
    if not valid:
        return None
    total_time = max(r.total_time for r in valid)
    total_tokens = sum(r.total_tokens for r in valid)
    return {
        "concurrency": concurrency,
        "throughput_tps": total_tokens / total_time,
        "avg_ttft": statistics.mean(r.ttft for r in valid),
        "avg_itl": statistics.mean(statistics.mean(r.itl_values) for r in valid if r.itl_values),
        "success_rate": len(valid) / len(results),
    }

async def benchmark(
    config: EngineConfig, concurrency_levels: list[int], requests_per_level: int
) -> list[dict[str, Any]]:
    """For each concurrency in ``concurrency_levels``, run ``requests_per_level`` requests.

    Prints a line per level. Returns only successful levels (skipped levels omit from list).
    """
    print(f"\n=== Benchmarking {config.name} ({config.base_url}) ===\n")
    results = []
    for c in concurrency_levels:
        print(f"  Concurrency {c:>3d}... ", end="", flush=True)
        r = await run_concurrency_level(config, c, requests_per_level)
        if r:
            results.append(r)
            print(f"throughput={r['throughput_tps']:.1f} tok/s, Avg TTFT={r['avg_ttft']*1000:.0f}ms, ITL={r['avg_itl']*1000:.1f}ms")
        else:
            print("FAILED")
    return results

def generate_chart(
    friendli_results: list[dict[str, Any]],
    vllm_results: list[dict[str, Any]],
    output_path: str = "benchmark_results.png",
) -> None:
    """Plot Friendli vs vLLM: throughput, average TTFT (ms), average ITL (ms) vs concurrency.

    Writes a PNG to ``output_path`` (dpi 150).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fc = [r["concurrency"] for r in friendli_results]
    vc = [r["concurrency"] for r in vllm_results]

    # Panel 1: Throughput
    axes[0].plot(fc, [r["throughput_tps"] for r in friendli_results],
                "o-", color="#2563EB", linewidth=2, markersize=7, label="Friendli Engine")
    axes[0].plot(vc, [r["throughput_tps"] for r in vllm_results],
                "s--", color="#DC2626", linewidth=2, markersize=7, label="vLLM")
    axes[0].set_xlabel("Concurrent Requests")
    axes[0].set_ylabel("Throughput (tokens/sec)")
    axes[0].set_title("Throughput vs Concurrency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(fc)

    # Panel 2: TTFT (avg)
    axes[1].plot(fc, [r["avg_ttft"] * 1000 for r in friendli_results],
                "o-", color="#2563EB", linewidth=2, markersize=7, label="Friendli Engine")
    axes[1].plot(vc, [r["avg_ttft"] * 1000 for r in vllm_results],
                "s--", color="#DC2626", linewidth=2, markersize=7, label="vLLM")
    axes[1].set_xlabel("Concurrent Requests")
    axes[1].set_ylabel("Avg TTFT (ms)")
    axes[1].set_title("Time to First Token vs Concurrency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(fc)

    # Panel 3: ITL (avg)
    axes[2].plot(fc, [r["avg_itl"] * 1000 for r in friendli_results],
                "o-", color="#2563EB", linewidth=2, markersize=7, label="Friendli Engine")
    axes[2].plot(vc, [r["avg_itl"] * 1000 for r in vllm_results],
                "s--", color="#DC2626", linewidth=2, markersize=7, label="vLLM")
    axes[2].set_xlabel("Concurrent Requests")
    axes[2].set_ylabel("Avg ITL (ms)")
    axes[2].set_title("Inter-Token Latency vs Concurrency")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(fc)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_path}")

async def main() -> None:
    """Entry point: parse arguments, run both benchmarks, print summary, save chart."""
    parser = argparse.ArgumentParser(description="Benchmark Friendli vs vLLM")
    parser.add_argument("--friendli-url", default="https://api.friendli.ai/dedicated/v1",
                        help="Friendli dedicated endpoint base URL (default: https://api.friendli.ai/dedicated/v1)")
    parser.add_argument("--friendli-model", required=True,
                        help="Friendli endpoint ID")
    parser.add_argument("--friendli-token", default=os.getenv("FRIENDLI_TOKEN", ""),
                        help="Friendli API token (or set FRIENDLI_TOKEN env var)")
    parser.add_argument("--vllm-url", required=True,
                        help="vLLM base URL (e.g., http://vllm-server:8000/v1)")
    parser.add_argument("--vllm-model", required=True,
                        help="vLLM model name (e.g., Qwen/Qwen3.5-9B)")
    parser.add_argument("--requests", type=int, default=64, help="Requests per concurrency level")
    parser.add_argument("--output", default="benchmark_results.png", help="Output chart path")
    args = parser.parse_args()

    friendli_config = EngineConfig(
        name="Friendli Engine",
        base_url=args.friendli_url,
        model=args.friendli_model,
        api_key=args.friendli_token,
    )
    vllm_config = EngineConfig(
        name="vLLM",
        base_url=args.vllm_url,
        model=args.vllm_model,
    )

    concurrency_levels = [1, 4, 16, 64]
    friendli = await benchmark(friendli_config, concurrency_levels, args.requests)
    vllm = await benchmark(vllm_config, concurrency_levels, args.requests)

    header = (f"{'Conc':>6} | {'Friendli tok/s':>14} {'Avg TTFT':>10} {'Avg ITL':>9}"
             f" | {'vLLM tok/s':>12} {'Avg TTFT':>10} {'Avg ITL':>9} | {'Speedup':>8}")
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for f, v in zip(friendli, vllm):
        speedup = f["throughput_tps"] / v["throughput_tps"] if v["throughput_tps"] > 0 else 0
        print(f"{f['concurrency']:>6d} | {f['throughput_tps']:>14.1f} {f['avg_ttft']*1000:>9.0f}ms {f['avg_itl']*1000:>8.1f}ms"
              f" | {v['throughput_tps']:>12.1f} {v['avg_ttft']*1000:>9.0f}ms {v['avg_itl']*1000:>8.1f}ms | {speedup:>7.2f}x")

    generate_chart(friendli, vllm, args.output)

if __name__ == "__main__":
    asyncio.run(main())




