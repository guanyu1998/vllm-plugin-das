"""Compare cached and cold continuations through a dedicated Qwen4Exp PD proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import time
import urllib.request
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_base")
    parser.add_argument("--metrics-url", action="append", required=True)
    parser.add_argument("--prefill-ranks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen38-next")
    parser.add_argument("--prompt-kind", choices=("text", "random"), default="text")
    parser.add_argument("--cold-rounds", type=int, default=2)
    parser.add_argument("--min-hit-tokens-per-rank", type=int, default=8000)
    parser.add_argument("--logprob-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.prefill_ranks < 1 or args.cold_rounds < 1:
        parser.error("prefill-ranks and cold-rounds must be positive")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(url, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"} if body else {},
        )
        with opener.open(req, timeout=1800) as response:
            return response.read()

    def hits():
        total = 0.0
        for url in args.metrics_url:
            text = request(url.rstrip("/") + "/metrics").decode()
            for line in text.splitlines():
                if re.match(r"vllm:prefix_cache_hits_total(?:\{|\s)", line):
                    total += float(line.split()[-1])
        return total

    rng = random.Random(20261118)
    if args.prompt_kind == "random":
        prefix = [rng.randrange(1000, 50000) for _ in range(8192)]
        producer = prefix + [rng.randrange(1000, 50000) for _ in range(37)]
        consumer = prefix + [rng.randrange(1000, 50000) for _ in range(91)]
    else:
        def tokenize(text, chat=False):
            payload = {"model": args.model, "prompt": text}
            if chat:
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": text}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            url = args.metrics_url[0].rstrip("/") + "/tokenize"
            return json.loads(request(url, payload))["tokens"]

        text = "Python reference examples for a programming exercise.\n\n"
        text += "\n".join(
            f"Example {index}: def add_{index}(left, right): return left + right\n"
            f"The function adds two integers; add_{index}({index}, 7) returns {index + 7}.\n"
            for index in range(500)
        )
        prefix = tokenize(text)[:8192]
        url = args.metrics_url[0].rstrip("/") + "/detokenize"
        text = json.loads(request(url, {"model": args.model, "tokens": prefix}))["prompt"]
        producer = tokenize(text +
            "\n\nNow write a Python function multiply(left, right). "
            "Return only Python source code, beginning with def.\n", chat=True,
        )
        consumer = tokenize(text +
            "\n\nNow write a Python function subtract(left, right) that returns "
            "the first integer minus the second integer. Return only Python "
            "source code, beginning with def.\n", chat=True,
        )
    shared_salt = uuid.uuid4().hex
    results = []

    def completion(phase, prompt, salt):
        started = time.monotonic()
        payload = {
            "model": args.model, "prompt": prompt, "cache_salt": salt,
            "max_tokens": 4, "temperature": 0, "seed": 0,
            "ignore_eos": True, "logprobs": 5, "return_token_ids": True,
        }
        response = json.loads(request(args.api_base.rstrip("/") + "/v1/completions", payload))
        result = {"phase": phase, "elapsed_seconds": time.monotonic() - started,
                  "response": response}
        results.append(result)
        print(phase, result["elapsed_seconds"], flush=True)
        return response["choices"][0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"requests": results, "prompt_kind": args.prompt_kind,
               "cold_rounds": args.cold_rounds,
               "logprob_tolerance": args.logprob_tolerance,
               "producer_tokens": len(producer), "consumer_tokens": len(consumer)}
    try:
        for rank in range(args.prefill_ranks):
            completion(f"producer_{rank}", producer, shared_salt)
        before = hits()
        cached = [completion(f"cached_{rank}", consumer, shared_salt)
                  for rank in range(args.prefill_ranks)]
        summary["prefix_hit_tokens_delta"] = hits() - before
        cold = []
        for round_index in range(args.cold_rounds):
            cold_salt = uuid.uuid4().hex
            cold.extend(completion(f"cold_{round_index}_{rank}", consumer, cold_salt)
                        for rank in range(args.prefill_ranks))
        max_difference = 0.0
        mismatches = []
        for index, cold_choice in enumerate(cold):
            cached_choice = cached[index % args.prefill_ranks]
            cached_ids, cold_ids = cached_choice.get("token_ids"), cold_choice.get("token_ids")
            if cached_ids is not None and cold_ids is not None:
                matches = cached_ids == cold_ids
            else:
                matches = cached_choice["text"] == cold_choice["text"]
            if not matches:
                mismatches.append(index)
                continue
            for cached_prob, cold_prob in zip(
                cached_choice["logprobs"]["token_logprobs"],
                cold_choice["logprobs"]["token_logprobs"],
            ):
                if cached_prob is not None and cold_prob is not None:
                    max_difference = max(max_difference, abs(cached_prob - cold_prob))
        summary["max_selected_token_logprob_difference"] = max_difference
        summary["cached_cold_token_mismatches"] = mismatches
        summary["cold_control_token_mismatches"] = [
            index for index, choice in enumerate(cold[args.prefill_ranks:])
            if choice["text"] != cold[index % args.prefill_ranks]["text"]
        ]
        required_hits = args.prefill_ranks * args.min_hit_tokens_per_rank
        assert summary["prefix_hit_tokens_delta"] >= required_hits, "Insufficient prefix hits"
        assert not mismatches, f"Cached/cold token mismatches: {mismatches}"
        assert max_difference <= args.logprob_tolerance, f"Logprob difference: {max_difference}"
        summary["passed"] = True
    except Exception as error:
        summary["passed"] = False
        summary["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
