import asyncio, time, json, os, argparse, statistics
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Install: pip install aiohttp")
    exit(1)

@dataclass
class RequestResult:
    ttft: float = 0.0
    itl_values: list = field(default_factory=list)
    total_tokens: int = 0
    total_time: float = 0.0
    error: Optional[str] = None

PROMPTS = [
    "Explain how KV caching works in transformer inference.",
    "Write a Python function to implement binary search.",
    "What are the trade-offs between tensor and pipeline parallelism?",
    "Describe the ORCA scheduling algorithm for LLM inference.",
    "Compare continuous batching with static batching.",
    "Explain PagedAttention and its memory management benefits.",
    "Write a Kubernetes StatefulSet YAML for GPU inference.",
    "What is speculative decoding and how does it reduce latency?",
]

@dataclass
class EngineConfig:
    """Configuration for an inference engine endpoint."""
    name: str
    base_url: str        # e.g., "https://api.friendli.ai/dedicated/v1"
    model: str           # e.g., "depnrsi6t162u7r" for Friendli, model name for vLLM
    api_key: str = ""    # Bearer token (required for Friendli, optional for vLLM)

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

async def send_streaming_request(session, config: EngineConfig, prompt):
    """Send a single streaming request and measure timing."""
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

async def run_concurrency_level(config, concurrency, num_requests):
    """Run num_requests at given concurrency, return aggregated results."""
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
        "p50_ttft": statistics.median(r.ttft for r in valid),
        "p99_ttft": sorted(r.ttft for r in valid)[int(len(valid)*0.99)],
        "avg_itl": statistics.mean(statistics.mean(r.itl_values) for r in valid if r.itl_values),
        "success_rate": len(valid) / len(results),
    }

async def benchmark_engine(config, concurrency_levels, requests_per_level):
    """Benchmark a single engine across all concurrency levels."""
    print(f"\n=== Benchmarking {config.name} ({config.base_url}) ===\n")
    results = []
    for c in concurrency_levels:
        print(f"  Concurrency {c:>3d}... ", end="", flush=True)
        r = await run_concurrency_level(config, c, requests_per_level)
        if r:
            results.append(r)
            print(f"throughput={r['throughput_tps']:.1f} tok/s, TTFT={r['avg_ttft']*1000:.0f}ms, TTFT p50={r['p50_ttft']*1000:.0f}ms, TTFT p99={r['p99_ttft']*1000:.0f}ms, ITL={r['avg_itl']*1000:.1f}ms")
        else:
            print("FAILED")
    return results

async def main():
    parser = argparse.ArgumentParser(description="Benchmark Friendli vs vLLM")
    parser.add_argument("--friendli-url", default="https://api.friendli.ai/dedicated/v1",
                        help="Friendli dedicated endpoint base URL (default: https://api.friendli.ai/dedicated/v1)")
    parser.add_argument("--friendli-model", required=True,
                        help="Friendli endpoint ID (e.g., depnrsi6t162u7r)")
    parser.add_argument("--friendli-token", default=os.getenv("FRIENDLI_TOKEN", ""),
                        help="Friendli API token (or set FRIENDLI_TOKEN env var)")
    parser.add_argument("--vllm-url", required=True,
                        help="vLLM base URL (e.g., http://vllm-server:8000/v1)")
    parser.add_argument("--vllm-model", required=True,
                        help="vLLM model name (e.g., Qwen/Qwen3.5-9B)")
    parser.add_argument("--requests", type=int, required=True,
                        help="Requests per concurrency level")
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
    friendli = await benchmark_engine(friendli_config, concurrency_levels, args.requests)
    vllm = await benchmark_engine(vllm_config, concurrency_levels, args.requests)

    print("\n" + "="*70)
    print(f"{'Concurrency':>12} {'Friendli tok/s':>16} {'vLLM tok/s':>14} {'Speedup':>10}")
    print("-"*70)
    for f, v in zip(friendli, vllm):
        speedup = f["throughput_tps"] / v["throughput_tps"] if v["throughput_tps"] > 0 else 0
        print(f"{f['concurrency']:>12d} {f['throughput_tps']:>16.1f} {v['throughput_tps']:>14.1f} {speedup:>9.2f}x")

if __name__ == "__main__":
    asyncio.run(main())



