#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import base64
import random
import asyncio
import statistics
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import aiohttp


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    latency_s: float
    status: int
    error: str = ""
    output_image_bytes: int = 0
    prompt: str = ""


def load_image_as_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_image_bytes(resp_json: Dict[str, Any]) -> bytes:
    """
    兼容典型 vLLM-Omni image edit 响应格式：
    choices[0].message.content[0].image_url.url = data:image/png;base64,...
    """
    choices = resp_json.get("choices", [])
    if not choices:
        raise ValueError("response has no choices")

    msg = choices[0].get("message", {})
    content = msg.get("content", [])

    if not isinstance(content, list):
        raise ValueError("message.content is not a list")

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "")
            if not url.startswith("data:"):
                raise ValueError("image_url.url is not a data URL")
            _, b64_data = url.split(",", 1)
            return base64.b64decode(b64_data)

    raise ValueError("no image_url found in response")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    image_data_urls: List[str],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    timeout_s: float,
    request_id: int,
    save_dir: Optional[str] = None,
    save_prob: float = 0.5,
    extra_top_level: Optional[Dict[str, Any]] = None,
) -> RequestResult:
    content = [{"type": "text", "text": prompt}]
    for img in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": img}})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }
    }

    if seed is not None:
        payload["extra_body"]["seed"] = seed

    if extra_top_level:
        payload.update(extra_top_level)

    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            status = resp.status
            text = await resp.text()
            latency_s = time.perf_counter() - t0

            if status != 200:
                return RequestResult(
                    request_id=request_id,
                    ok=False,
                    latency_s=latency_s,
                    status=status,
                    error=text[:1000],
                    prompt=prompt,
                )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return RequestResult(
                    request_id=request_id,
                    ok=False,
                    latency_s=latency_s,
                    status=status,
                    error=f"invalid json: {e}; body={text[:1000]}",
                    prompt=prompt,
                )

            try:
                img_bytes = extract_image_bytes(data)
            except Exception as e:
                return RequestResult(
                    request_id=request_id,
                    ok=False,
                    latency_s=latency_s,
                    status=status,
                    error=f"parse image failed: {e}",
                    prompt=prompt,
                )

            if save_dir and random.random() < save_prob:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"resp_{request_id:06d}.png")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

            return RequestResult(
                request_id=request_id,
                ok=True,
                latency_s=latency_s,
                status=status,
                output_image_bytes=len(img_bytes),
                prompt=prompt,
            )

    except asyncio.TimeoutError:
        latency_s = time.perf_counter() - t0
        return RequestResult(
            request_id=request_id,
            ok=False,
            latency_s=latency_s,
            status=0,
            error=f"timeout>{timeout_s}s",
            prompt=prompt,
        )
    except Exception as e:
        latency_s = time.perf_counter() - t0
        return RequestResult(
            request_id=request_id,
            ok=False,
            latency_s=latency_s,
            status=0,
            error=str(e),
            prompt=prompt,
        )


async def warmup(
    session: aiohttp.ClientSession,
    args,
    image_data_urls: List[str],
    prompts: List[str],
):
    if args.warmup_requests <= 0:
        return

    print(f"[warmup] start, requests={args.warmup_requests}")
    for i in range(args.warmup_requests):
        prompt = prompts[i % len(prompts)]
        result = await one_request(
            session=session,
            url=args.url,
            model=args.model,
            prompt=prompt,
            image_data_urls=image_data_urls,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            timeout_s=args.timeout,
            request_id=-(i + 1),
            save_dir=None,
            save_prob=0.0,
            extra_top_level=None,
        )
        state = "ok" if result.ok else "fail"
        print(f"[warmup] {i+1}/{args.warmup_requests} {state} latency={result.latency_s:.3f}s status={result.status}")
    print("[warmup] done")


async def run_fixed_concurrency(
    session: aiohttp.ClientSession,
    args,
    image_data_urls: List[str],
    prompts: List[str],
) -> List[RequestResult]:
    results: List[RequestResult] = []
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []

    async def runner(req_id: int):
        async with sem:
            prompt = prompts[req_id % len(prompts)]
            return await one_request(
                session=session,
                url=args.url,
                model=args.model,
                prompt=prompt,
                image_data_urls=image_data_urls,
                height=args.height,
                width=args.width,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=(args.seed + req_id) if args.seed is not None and args.vary_seed else args.seed,
                timeout_s=args.timeout,
                request_id=req_id,
                save_dir=args.save_dir,
                save_prob=args.save_prob,
                extra_top_level=None,
            )

    t0 = time.perf_counter()
    for req_id in range(args.num_requests):
        tasks.append(asyncio.create_task(runner(req_id)))

    for fut in asyncio.as_completed(tasks):
        res = await fut
        results.append(res)
        if args.progress_every > 0 and len(results) % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"[progress] done={len(results)}/{args.num_requests}, elapsed={elapsed:.1f}s")

    return results


async def run_fixed_rate(
    session: aiohttp.ClientSession,
    args,
    image_data_urls: List[str],
    prompts: List[str],
) -> List[RequestResult]:
    """
    类似 bench serve 的 request-rate 模式：
    按固定 RPS 发起请求，不限制客户端最大并发，最后统计真实峰值由服务端决定。
    """
    results: List[RequestResult] = []
    tasks = []

    if args.request_rate <= 0:
        raise ValueError("request_rate must be > 0 in fixed-rate mode")

    interval = 1.0 / args.request_rate

    async def runner(req_id: int):
        prompt = prompts[req_id % len(prompts)]
        return await one_request(
            session=session,
            url=args.url,
            model=args.model,
            prompt=prompt,
            image_data_urls=image_data_urls,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=(args.seed + req_id) if args.seed is not None and args.vary_seed else args.seed,
            timeout_s=args.timeout,
            request_id=req_id,
            save_dir=args.save_dir,
            save_prob=args.save_prob,
            extra_top_level=None,
        )

    start_time = time.perf_counter()
    for req_id in range(args.num_requests):
        now = time.perf_counter()
        target = start_time + req_id * interval
        if target > now:
            await asyncio.sleep(target - now)
        tasks.append(asyncio.create_task(runner(req_id)))

    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
        if args.progress_every > 0 and len(results) % args.progress_every == 0:
            print(f"[progress] done={len(results)}/{args.num_requests}")

    return results


def summarize(results: List[RequestResult], total_elapsed_s: float) -> Dict[str, Any]:
    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in ok_results]
    out_bytes = [r.output_image_bytes for r in ok_results]

    summary = {
        "total_requests": len(results),
        "successful_requests": len(ok_results),
        "failed_requests": len(fail_results),
        "success_rate": (len(ok_results) / len(results) * 100.0) if results else 0.0,
        "benchmark_duration_s": total_elapsed_s,
        "request_throughput_req_s": (len(ok_results) / total_elapsed_s) if total_elapsed_s > 0 else 0.0,
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_s": statistics.median(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "avg_output_image_kb": (statistics.mean(out_bytes) / 1024.0) if out_bytes else 0.0,
    }
    return summary


def print_summary(summary: Dict[str, Any], results: List[RequestResult]):
    print("\n============ Image-to-Image Serving Benchmark Result ============")
    print(f"Total requests:            {summary['total_requests']}")
    print(f"Successful requests:       {summary['successful_requests']}")
    print(f"Failed requests:           {summary['failed_requests']}")
    print(f"Success rate:              {summary['success_rate']:.2f}%")
    print(f"Benchmark duration (s):    {summary['benchmark_duration_s']:.2f}")
    print(f"Request throughput (req/s):{summary['request_throughput_req_s']:.2f}")
    print("--------------- Latency ---------------")
    print(f"Mean latency (s):          {summary['avg_latency_s']:.3f}")
    print(f"Median latency (s):        {summary['median_latency_s']:.3f}")
    print(f"P50 latency (s):           {summary['p50_latency_s']:.3f}")
    print(f"P95 latency (s):           {summary['p95_latency_s']:.3f}")
    print(f"P99 latency (s):           {summary['p99_latency_s']:.3f}")
    print("--------------- Output Image ---------------")
    print(f"Avg output image (KB):     {summary['avg_output_image_kb']:.2f}")

    failed = [r for r in results if not r.ok]
    if failed:
        print("--------------- Fail Samples ---------------")
        for r in failed[:10]:
            print(f"[id={r.request_id}] status={r.status} latency={r.latency_s:.3f}s error={r.error[:300]}")


def load_prompts(args) -> List[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [x.strip() for x in f if x.strip()]
        if not prompts:
            raise ValueError("prompt_file is empty")
        return prompts

    if args.prompt:
        return [args.prompt]

    return [
        "Convert this image to watercolor style",
        "Turn this image into a realistic oil painting",
        "Make this image look cinematic with richer lighting",
        "Transform this image into an anime illustration",
        "Convert this image into a pixel-art style",
    ]


async def main_async(args):
    image_data_urls = [load_image_as_data_url(p) for p in args.input_images]
    prompts = load_prompts(args)

    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        await warmup(session, args, image_data_urls, prompts)

        t0 = time.perf_counter()
        if args.mode == "concurrency":
            results = await run_fixed_concurrency(session, args, image_data_urls, prompts)
        else:
            results = await run_fixed_rate(session, args, image_data_urls, prompts)
        total_elapsed_s = time.perf_counter() - t0

    summary = summarize(results, total_elapsed_s)
    print_summary(summary, results)

    if args.output_json:
        out = {
            "config": vars(args),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report to: {args.output_json}")


def build_parser():
    p = argparse.ArgumentParser(description="Bench vLLM-Omni image-to-image serving performance")
    p.add_argument("--url", default="http://127.0.0.1:8092/v1/chat/completions")
    p.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    p.add_argument("--api-key", default="none")

    p.add_argument("--input-images", nargs="+", required=True,
                   help="One or more input images. For multi-image edit, pass multiple files.")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None,
                   help="Text file, one prompt per line")

    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vary-seed", action="store_true",
                   help="Use seed + request_id to diversify each request")

    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup-requests", type=int, default=2)

    p.add_argument("--mode", choices=["concurrency", "rate"], default="concurrency")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-rate", type=float, default=1.0,
                   help="Requests per second, only used in rate mode")
    p.add_argument("--num-requests", type=int, default=20)

    p.add_argument("--save-dir", default=None,
                   help="Optionally save some response images")
    p.add_argument("--save-prob", type=float, default=0.0,
                   help="Probability to save each successful response image")
    p.add_argument("--output-json", default="bench_i2i_report.json")
    p.add_argument("--progress-every", type=int, default=10)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.prompt and args.prompt_file:
        print("Use either --prompt or --prompt-file, not both.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "concurrency" and args.concurrency <= 0:
        print("--concurrency must be > 0", file=sys.stderr)
        sys.exit(1)

    if args.mode == "rate" and args.request_rate <= 0:
        print("--request-rate must be > 0", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
