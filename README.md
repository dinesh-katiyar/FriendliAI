# Inference Engine Benchmark: FriendliAI vs vLLM

Benchmarks FriendliAI Dedicated Endpoints against vLLM serving Qwen/Qwen3.5-9B, measuring throughput, time-to-first-token (TTFT), and inter-token latency (ITL) across concurrency levels.

## Why These Metrics

Throughput, TTFT, and ITL are the three metrics that matter most for LLM inference serving, and together they capture the full user experience of a streaming response.

**Throughput (tokens/sec)** measures aggregate serving capacity. For applications that dispatch many requests per session, throughput at high concurrency directly determines how many users the system can serve simultaneously. An engine that maintains high throughput as concurrency scales is more cost-efficient because it serves more tokens per GPU-second.

**Avg TTFT (ms)** measures perceived responsiveness. This is the time the user stares at a blank screen before the first token appears. For interactive use cases (chat, code completion, tool calls), TTFT under 500ms feels instant while TTFT over 2 seconds feels broken. TTFT is dominated by prefill computation (processing the input prompt through the model) and reveals how well each engine handles prompt processing under load.

**Avg ITL (ms)** measures streaming smoothness. Once tokens start flowing, ITL determines whether the output feels like fluid text or stuttering chunks. For coding agents that parse streaming tool calls, high ITL adds latency to every tool-use round trip. ITL is dominated by decode-step scheduling and memory bandwidth, which are different bottlenecks from prefill (TTFT). Measuring both separately reveals whether an engine is bottlenecked on prefill, decode, or both.

## Metrics

- **Throughput** (tokens/sec): Total tokens generated divided by wall-clock time at each concurrency level
- **Avg TTFT** (ms): Average time from request sent to first token received
- **Avg ITL** (ms): Average time between consecutive tokens in the stream

## Prerequisites

```bash
pip install aiohttp matplotlib
```

## Setup

### FriendliAI Dedicated Endpoint

You need an active FriendliAI dedicated endpoint. Get your API token and endpoint ID from the [FriendliAI console](https://suite.friendli.ai/).

```bash
export FRIENDLI_TOKEN=your_friendli_api_token
```

### vLLM

A running vLLM server with an OpenAI-compatible API serving Qwen3.5-9B:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --port 8000
```

## Usage

```bash
python benchmark_engines.py \
  --friendli-model friendly_endpoint_ID\
  --vllm-url http://localhost:8000/v1 \
  --vllm-model Qwen/Qwen3.5-9B \
  --requests 64
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--friendli-model` | Yes | - | Friendli endpoint ID  |
| `--friendli-url` | No | `https://api.friendli.ai/dedicated/v1` | Friendli base URL |
| `--friendli-token` | No | `FRIENDLI_TOKEN` env var | Friendli API token |
| `--vllm-url` | Yes | - | vLLM base URL (e.g., `http://localhost:8000/v1`) |
| `--vllm-model` | Yes | - | vLLM model name (e.g., `Qwen/Qwen3.5-9B`) |
| `--requests` | No | `64` | Requests per concurrency level |
| `--output` | No | `benchmark_results.png` | Output chart path |

## Output

### Console

The script prints progress as each concurrency level completes, followed by a summary table:

```
=== Benchmarking Friendli Engine (https://api.friendli.ai/dedicated/v1) ===

  Concurrency   1... throughput=42.3 tok/s, Avg TTFT=187ms, ITL=23.4ms
  Concurrency   4... throughput=156.8 tok/s, Avg TTFT=203ms, ITL=25.1ms
  Concurrency  16... throughput=589.2 tok/s, Avg TTFT=245ms, ITL=27.8ms
  Concurrency  64... throughput=1823.5 tok/s, Avg TTFT=412ms, ITL=35.2ms

  Conc | Friendli tok/s   Avg TTFT   Avg ITL |   vLLM tok/s   Avg TTFT   Avg ITL |  Speedup
-------...
     1 |           42.3      187ms     23.4ms |         38.1      142ms     26.1ms |    1.11x
     4 |          156.8      203ms     25.1ms |        128.4      165ms     28.9ms |    1.22x
    16 |          589.2      245ms     27.8ms |        412.6      298ms     33.4ms |    1.43x
    64 |         1823.5      412ms     35.2ms |        987.3      687ms     48.7ms |    1.85x
```

*(Sample output. Actual numbers depend on hardware and network.)*

### Chart

A 3-panel PNG comparing both engines across concurrency levels:

| Panel | Metric | Y-axis |
|---|---|---|
| Left | Throughput | tokens/sec |
| Center | Avg TTFT | ms |
| Right | Avg ITL | ms |

Saved to `benchmark_results.png` (or path specified by `--output`).

## How It Works

The benchmark uses `aiohttp` for raw async HTTP requests (faster than httpx/OpenAI SDK at high concurrency due to C-based HTTP parser). For each concurrency level:

1. Launches N total requests (set by `--requests`)
2. Uses an `asyncio.Semaphore` to limit concurrent in-flight requests
3. Parses SSE streaming responses to measure per-token timing
4. Aggregates results: throughput, avg TTFT, avg ITL, success rate

Both engines are called via the OpenAI-compatible `/v1/chat/completions` endpoint with streaming enabled.

## Notes

- The script cycles through 8 built-in prompts. All prompts are similar length to keep the comparison consistent.
- Each request generates up to 256 tokens (`max_tokens=256`).
