# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Mooncake PD 请求路由 proxy（HCU 版）。

源自 vLLM 的 ``examples/disaggregated/mooncake_connector/
mooncake_connector_proxy.py``（v0.28.1-dev，commit 29071fe0e8）。
vLLM 本体不做改动，所有修复在本副本中实现。

相对上游的改动（见 qwen3_8-next-flash-PD/bug_record/BUG-002）：

1. ``_report_prefill_failure()``：prefill 腿是 fire-and-forget task，
   其异常原本无人 retrieve。任何失败（连接被重置、非 2xx、后端已死）
   都没有任何日志，唯一可见症状是 ``VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT``
   秒之后 P 侧报 "Timeout waiting for P side ready"。现在立即打印。

2. ``_stream_guarded_by_prefill()``：prefill 腿失败时立刻中止 decode 腿
   并关闭其流。此前 D 会一直挂在 ``WAITING_FOR_REMOTE_KVS`` 等一个永不
   到来的 KV，等满 abort 超时后在 ``kv_load_failure_policy=fail`` 下把
   整个 D 引擎打死（4 rank 全灭、栈需全重启）。实测：挂起请求在守卫
   触发后约 2s 内被释放，故障从"栈级毁坏"降级为"单请求失败"。

3. ``get_prefiller_info()``：bootstrap ``/query`` 与 ``/health`` 一起重试，
   并拒绝空的 dp_rank map。此前 ``/query`` 的瞬时错误会让 proxy 永久
   停在未就绪状态，所有请求返回 503，只能手工重启 proxy。

4. 诊断 ``print()`` 全部加 ``flush=True``：经 ``tee`` 落盘时 stdout 是块
   缓冲的，否则故障现场要等缓冲区满才可见。配合启动脚本的 ``python3 -u``。

注意：proxy 启动时会缓存各 P 的 engine_id，**P 重启后必须重启本 proxy**。
另外起栈前务必确认目标端口没有残留的旧实例前端——vLLM 的 API server 用
SO_REUSEPORT，残留前端会与新实例一起被内核分发连接，且照常回 /health 200
（见 BUG-002 §9.7）。
"""

import argparse
import asyncio
import ipaddress
import itertools
import os
import urllib
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def make_http_path(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def prefiller_cycle(prefill_clients: list[Any]):
    while True:
        for prefill_client in prefill_clients:
            for i in range(prefill_client["dp_size"]):
                yield prefill_client, i


async def get_prefiller_info(prefill_clients: list, ready: asyncio.Event):
    for prefill_client in prefill_clients:
        while True:
            try:
                # Wait for prefill service to be ready
                response = await prefill_client["client"].get("/health")
                response.raise_for_status()

                # The bootstrap /query has to be retried together with
                # /health: the bootstrap server and the API server come up at
                # different times, so a transient error here used to leave the
                # proxy permanently un-ready (every request answered 503 until
                # the proxy was restarted by hand).
                response = await prefill_client["client"].get(
                    prefill_client["bootstrap_addr"] + "/query"
                )
                response.raise_for_status()
                data = response.json()
                if not data:
                    raise RuntimeError(
                        f"bootstrap {prefill_client['bootstrap_addr']} returned "
                        "an empty dp_rank map"
                    )
            except Exception as e:
                print(f"Waiting for prefiller {prefill_client['url']}: {e!r}", flush=True)
                await asyncio.sleep(1)
                continue
            break

        for dp_rank, dp_entry in data.items():
            prefill_client["dp_engine_id"][int(dp_rank)] = dp_entry["engine_id"]
        dp_size = len(data)
        prefill_client["dp_size"] = dp_size
        print(f"Inited prefiller {prefill_client['url']} with dp_size={dp_size}", flush=True)

    ready.set()
    print("All prefiller instances are ready.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.ready = asyncio.Event()

    # Create prefill clients
    for i, (url, bootstrap_port) in enumerate(global_args.prefill):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "url": url,
                "bootstrap_addr": make_http_path(hostname, bootstrap_port or 8998),
                "dp_engine_id": {},
            }
        )

    # Create decode clients
    for i, url in enumerate(global_args.decode):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
            }
        )

    asyncio.create_task(get_prefiller_info(app.state.prefill_clients, app.state.ready))

    # Initialize round-robin iterators
    app.state.prefill_iterator = prefiller_cycle(app.state.prefill_clients)
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Got {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "bootstrap_port"),
        help=(
            "Prefill server URL and optional bootstrap port. "
            "Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, "
            "'none', or omitted (defaults to none)."
        ),
    )

    # For decoder instances
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar=("URL",),
        help="Decode server URL. Can be specified multiple times.",
    )

    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)

    return args


# From sglang router_args.py
def _parse_prefill_urls(prefill_list):
    """Parse prefill URLs from --prefill arguments.

    Format: --prefill URL [BOOTSTRAP_PORT]
    Example:
        --prefill http://prefill1:8080 9000  # With bootstrap port
        --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
        --prefill http://prefill3:8080       # Defaults to no bootstrap port
    """
    if not prefill_list:
        return []

    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]

        # Handle optional bootstrap port
        if len(prefill_args) >= 2:
            bootstrap_port_str = prefill_args[1]
            # Handle 'none' as None
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"  # noqa: E501
                    ) from e
        else:
            # No bootstrap port specified, default to None
            bootstrap_port = None

        prefill_urls.append((url, bootstrap_port))

    return prefill_urls


def _parse_decode_urls(decode_list):
    """Parse decode URLs from --decode arguments.

    Format: --decode URL
    Example: --decode http://decode1:8081 --decode http://decode2:8081
    """
    if not decode_list:
        return []

    # decode_list is a list of single-element lists due to nargs=1
    return [url[0] for url in decode_list]


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        return next(app.state.prefill_iterator)
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, dp_rank: int, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"xfer-{request_id}",
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # CRITICAL: Release connection back to pool
    await response.aclose()


async def stream_service_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    req_data["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }

    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def _report_prefill_failure(task: asyncio.Task, request_id: str) -> None:
    """Make every prefill failure visible, whoever notices it first.

    A prefill POST that is dropped (connection reset while a producer restarts,
    non-2xx, ...) leaves no trace of its own: the task is fire-and-forget, and
    if the decode leg happens to finish first (full prefix-cache hit on D) even
    the asyncio "Task exception was never retrieved" message is the only hint.
    """

    def _report(done: asyncio.Task) -> None:
        if done.cancelled():
            return
        exc = done.exception()
        if exc is not None:
            print(f"Prefill leg failed for request {request_id}: {exc!r}", flush=True)

    task.add_done_callback(_report)


async def _stream_guarded_by_prefill(
    prefill_task: asyncio.Task,
    chunks,
    request_id: str,
):
    """Relay decode chunks, but fail fast if the prefill leg died.

    The prefill request is fired as a background task so it can overlap with
    the decode leg parking itself on the producer's KV. Without this guard a
    failed prefill POST is completely silent: the decoder waits out
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT (480s by default) for KV that will
    never be produced, the producer logs "Timeout waiting for P side ready",
    and with kv_load_failure_policy=fail that timeout kills the whole decode
    engine. Closing the decode stream here also lets the decoder abort the
    request instead of holding its blocks. See bug_record/BUG-002.
    """
    agen = chunks.__aiter__()
    chunk_task: asyncio.Future | None = None
    try:
        while True:
            if chunk_task is None:
                chunk_task = asyncio.ensure_future(agen.__anext__())
            waiters = {chunk_task}
            if not prefill_task.done():
                waiters.add(prefill_task)
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

            if prefill_task.done() and not prefill_task.cancelled():
                prefill_exc = prefill_task.exception()
                if prefill_exc is not None:
                    raise RuntimeError(
                        f"prefill leg failed for request {request_id}: "
                        f"{prefill_exc!r}"
                    ) from prefill_exc

            if chunk_task.done():
                try:
                    chunk = chunk_task.result()
                except StopAsyncIteration:
                    return
                finally:
                    chunk_task = None
                yield chunk
    finally:
        if chunk_task is not None and not chunk_task.done():
            chunk_task.cancel()
            with suppress(BaseException):
                await chunk_task
        with suppress(BaseException):
            await agen.aclose()


async def _handle_completions(api: str, request: Request):
    if not app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")

    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Get the next prefill client in round-robin fashion
        prefill_client_info, prefill_dp_rank = get_next_client(request.app, "prefill")

        # Send request to prefill service. Fire-and-forget on purpose (it must
        # overlap with the decode leg), but the task is handed to
        # _stream_guarded_by_prefill() so a failure cannot go unnoticed.
        prefill_task = asyncio.create_task(
            send_request_to_service(
                prefill_client_info, prefill_dp_rank, api, req_data, request_id
            )
        )
        _report_prefill_failure(prefill_task, request_id)

        decode_client_info = get_next_client(request.app, "decode")

        # Stream response from decode service
        async def generate_stream():
            async for chunk in _stream_guarded_by_prefill(
                prefill_task,
                stream_service_response(
                    prefill_client_info,
                    prefill_dp_rank,
                    decode_client_info,
                    api,
                    req_data,
                    request_id=request_id,
                ),
                request_id,
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint", flush=True)
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
