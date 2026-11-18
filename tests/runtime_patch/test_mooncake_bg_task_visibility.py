# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mooncake PD failure-visibility contracts (model-free).

Regression cover for BUG-002 (qwen3_8-next-flash-PD): a KV transfer that never
completes used to be invisible until the *peer* timed out
VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT seconds later, and a stalled connector event
loop used to block the engine step forever.

The three contracts pinned here are:

1. ``get_finished()`` polling never blocks the engine step indefinitely, and a
   poll that times out is resumed rather than abandoned (abandoning it would
   silently drop finished-sending/recving reports and strand KV blocks).
2. A fire-and-forget connector coroutine reports its exception. A
   ``concurrent.futures.Future`` from ``run_coroutine_threadsafe()`` prints
   nothing at all when its exception is never retrieved.
3. The proxy's prefill leg cannot fail silently: a dropped prefill POST aborts
   the decode leg instead of letting the decoder wait out the KV timeout and
   die under ``kv_load_failure_policy=fail``. vLLM stays unmodified, so the
   proxy under test is this repo's
   ``examples/disaggregated_serving_mooncake/mooncake_connector_proxy_hcu.py``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest


mooncake_connector = pytest.importorskip(
    "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
)


@pytest.fixture()
def bg_loop():
    """A connector-style event loop running in its own daemon thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


class _PollHarness:
    """Minimal stand-in exposing only the poll plumbing under test."""

    _POLL_TIMEOUT_S = 0.2
    _POLL_STALL_LOG_INTERVAL_S = 10_000.0  # never warn during tests
    tp_rank = 0

    _submit_poll = mooncake_connector.MooncakeConnectorWorker._submit_poll
    _collect_poll = mooncake_connector.MooncakeConnectorWorker._collect_poll

    def __init__(self) -> None:
        self._pending_polls: dict = {}
        self._poll_started_at: dict = {}
        self._poll_stall_logged_at: dict = {}


def _collect_until(harness: _PollHarness, name: str, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = harness._collect_poll(name)
        if result is not None:
            return result
    return None


def test_stalled_poll_does_not_block_step_and_keeps_its_result(bg_loop) -> None:
    harness = _PollHarness()
    released = asyncio.Event()
    starts = []

    async def slow_poll() -> set[str]:
        starts.append(1)
        await released.wait()
        return {"req-1"}

    harness._submit_poll("send", slow_poll, bg_loop)

    # The engine step gives up for this round instead of blocking forever.
    started = time.monotonic()
    assert harness._collect_poll("send") is None
    assert time.monotonic() - started < 2.0

    # The in-flight poll is retained, so the next step does not pile on a
    # second coroutine competing for the same destructive set swap.
    retained = harness._pending_polls["send"]
    harness._submit_poll("send", slow_poll, bg_loop)
    assert harness._pending_polls["send"] is retained
    assert len(starts) == 1

    # Once the loop makes progress the completion is reported, not lost.
    bg_loop.call_soon_threadsafe(released.set)
    assert _collect_until(harness, "send") == {"req-1"}
    assert harness._pending_polls == {}
    assert harness._poll_stall_logged_at == {}


def test_poll_failure_is_raised_once_then_cleared(bg_loop) -> None:
    harness = _PollHarness()

    async def failing_poll() -> set[str]:
        raise RuntimeError("Mooncake KV receive failed for ['req-1']")

    harness._submit_poll("recv", failing_poll, bg_loop)

    with pytest.raises(RuntimeError, match="Mooncake KV receive failed"):
        for _ in range(50):
            harness._collect_poll("recv")

    # The future is dropped so the same error is not re-raised every step.
    assert harness._pending_polls == {}
    assert harness._collect_poll("recv") is None


def test_fire_and_forget_coroutine_reports_its_exception(
    bg_loop, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[tuple] = []
    monkeypatch.setattr(
        mooncake_connector,
        "logger",
        SimpleNamespace(error=lambda *args, **kwargs: recorded.append(args)),
    )

    async def boom() -> None:
        raise KeyError("xfer-deadbeef")

    future = mooncake_connector._watch_bg_coroutine(
        asyncio.run_coroutine_threadsafe(boom(), bg_loop),
        "record_send_reqs",
    )

    with pytest.raises(KeyError):
        future.result(timeout=5)

    deadline = time.monotonic() + 5
    while not recorded and time.monotonic() < deadline:
        time.sleep(0.01)

    assert recorded, "a failed background coroutine must be logged"
    assert "record_send_reqs" in recorded[0]


class _RecordSendHarness:
    """Stand-in exercising only record_send_reqs() bookkeeping."""

    _pcp_dummy_sender = False
    record_send_reqs = mooncake_connector.MooncakeConnectorWorker.record_send_reqs

    def __init__(self) -> None:
        self.reqs_need_send: dict = {}
        self.finished_sending_reqs: set[str] = set()


def test_record_send_reqs_frees_blocks_when_entry_vanished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped send entry must not pin the producer's KV blocks forever.

    The lookup used to be an unguarded ``self.reqs_need_send[transfer_id]``;
    the resulting KeyError was swallowed by the fire-and-forget coroutine, so
    the request was never reported as finished-sending and its blocks stayed
    pinned while the consumer waited out the KV abort timeout.
    """
    warnings: list[tuple] = []
    monkeypatch.setattr(
        mooncake_connector,
        "logger",
        SimpleNamespace(
            warning=lambda *args, **kwargs: warnings.append(args),
            debug=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
    )

    harness = _RecordSendHarness()
    metadata = SimpleNamespace(
        reqs_to_send={"req-1": ("xfer-gone", [[1, 2, 3]])},
        reqs_not_processed=set(),
    )

    asyncio.run(harness.record_send_reqs(metadata))

    assert harness.finished_sending_reqs == {"req-1"}
    assert warnings, "the vanished send entry must be logged"
    assert "xfer-gone" in str(warnings[0])


def test_record_send_reqs_marks_ready_on_the_happy_path() -> None:
    harness = _RecordSendHarness()
    send_meta = mooncake_connector.SendBlockMeta(
        p_req_id="",
        transfer_id="xfer-1",
        local_block_ids=[],
        ready=asyncio.Event(),
    )
    harness.reqs_need_send["xfer-1"] = send_meta
    metadata = SimpleNamespace(
        reqs_to_send={"req-1": ("xfer-1", [[7, 8]])},
        reqs_not_processed=set(),
    )

    asyncio.run(harness.record_send_reqs(metadata))

    assert send_meta.ready.is_set()
    assert send_meta.p_req_id == "req-1"
    assert send_meta.local_block_ids == [[7, 8]]
    assert harness.finished_sending_reqs == set()


class _PullResultHarness:
    """Stand-in exercising only process_pulling_result()/note_finished_requests."""

    is_kv_producer = False
    _ABANDONED_PRUNE_AT = (
        mooncake_connector.MooncakeConnectorWorker._ABANDONED_PRUNE_AT
    )
    _ABANDONED_MAX = mooncake_connector.MooncakeConnectorWorker._ABANDONED_MAX
    note_finished_requests = (
        mooncake_connector.MooncakeConnectorWorker.note_finished_requests
    )
    process_pulling_result = (
        mooncake_connector.MooncakeConnectorWorker.process_pulling_result
    )
    _abandoned_retention_s = (
        mooncake_connector.MooncakeConnectorWorker._abandoned_retention_s
    )

    def __init__(self) -> None:
        self._abandoned_pull_reqs: dict = {}
        self.finished_recving_reqs: set[str] = set()
        self._kv_load_error: str | None = None


def _pull_meta(req_id: str, *, blocks):
    return mooncake_connector.PullReqMeta(
        d_req_id=req_id,
        transfer_id=f"xfer-{req_id}",
        local_block_ids=blocks,
        remote_engine_id="engine-0",
        remote_bootstrap_addr="http://127.0.0.1:18998",
        pull_tasks_count=1,
    )


def _error_response(err_reqs):
    return mooncake_connector.MooncakeXferResponse(
        status=mooncake_connector.MooncakeXferResponseStatus.FINISH,
        err_reqs=list(err_reqs),
        err_msg="Timeout waiting for P side ready.",
    )


def test_live_kv_receive_error_still_fails_the_worker() -> None:
    """Regression guard: a real failure must stay fatal under policy=fail."""
    harness = _PullResultHarness()
    metas = {"req-live": _pull_meta("req-live", blocks=[[1, 2]])}

    harness.process_pulling_result(_error_response(["req-live"]), metas)

    assert harness._kv_load_error is not None
    assert "req-live" in harness._kv_load_error
    assert harness.finished_recving_reqs == set()


def test_late_error_for_aborted_request_is_not_fatal_and_frees_blocks() -> None:
    """The BUG-002 delayed bomb: abort now, peer reports the timeout later.

    The consumer request was parked in WAITING_FOR_REMOTE_KVS and aborted, so
    the scheduler is holding its blocks until the connector reports the recv as
    finished. The producer only answers after
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT; that answer must neither kill the
    worker nor leave the blocks pinned.
    """
    harness = _PullResultHarness()
    harness.note_finished_requests({"req-aborted"})
    metas = {"req-aborted": _pull_meta("req-aborted", blocks=[[7, 8, 9]])}

    harness.process_pulling_result(_error_response(["req-aborted"]), metas)

    assert harness._kv_load_error is None
    assert harness.finished_recving_reqs == {"req-aborted"}
    assert metas["req-aborted"].pull_tasks_count == 0


def test_late_error_for_notify_and_free_pull_is_not_reported() -> None:
    """A pull with no local blocks belongs to an already-released request.

    Reporting it as finished-recving would hand the scheduler a request id it
    has deleted (``assert req_id in self.requests``).
    """
    harness = _PullResultHarness()
    harness.note_finished_requests({"req-unscheduled"})
    metas = {"req-unscheduled": _pull_meta("req-unscheduled", blocks=[[], []])}

    harness.process_pulling_result(_error_response(["req-unscheduled"]), metas)

    assert harness._kv_load_error is None
    assert harness.finished_recving_reqs == set()


def test_unattributable_error_id_is_not_fatal() -> None:
    """An error id that matches no local pull cannot be blamed on live traffic."""
    harness = _PullResultHarness()
    metas = {"req-known": _pull_meta("req-known", blocks=[[3]])}

    harness.process_pulling_result(_error_response(["9bb417933494c881"]), metas)

    assert harness._kv_load_error is None
    assert harness.finished_recving_reqs == set()


def test_mixed_errors_fail_only_the_live_request() -> None:
    harness = _PullResultHarness()
    harness.note_finished_requests({"req-aborted"})
    metas = {
        "req-aborted": _pull_meta("req-aborted", blocks=[[1]]),
        "req-live": _pull_meta("req-live", blocks=[[2]]),
    }

    harness.process_pulling_result(
        _error_response(["req-aborted", "req-live"]), metas
    )

    assert harness._kv_load_error is not None
    assert "req-live" in harness._kv_load_error
    assert "req-aborted" not in harness._kv_load_error
    assert harness.finished_recving_reqs == {"req-aborted"}


def test_abandoned_map_is_bounded() -> None:
    """A long-lived decode node must not accumulate abandoned ids forever."""
    harness = _PullResultHarness()
    cap = mooncake_connector.MooncakeConnectorWorker._ABANDONED_MAX
    harness.note_finished_requests({f"req-{i}" for i in range(cap + 5_000)})
    assert len(harness._abandoned_pull_reqs) <= cap
    # A freshly finished request must still be recognised after the prune.
    harness.note_finished_requests({"req-newest"})
    assert "req-newest" in harness._abandoned_pull_reqs


def test_producer_does_not_track_abandoned_requests() -> None:
    harness = _PullResultHarness()
    harness.is_kv_producer = True
    harness.note_finished_requests({"req-x"})
    assert harness._abandoned_pull_reqs == {}


def _load_proxy_module():
    """Load the HCU mooncake proxy shipped by *this* repo.

    vLLM itself must stay unmodified, so the BUG-002 proxy fixes live in
    ``examples/disaggregated_serving_mooncake/mooncake_connector_proxy_hcu.py``
    here rather than in vLLM's own example. Resolve it relative to this file so
    the test does not depend on the checkout location.
    """
    proxy_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disaggregated_serving_mooncake"
        / "mooncake_connector_proxy_hcu.py"
    )
    if not proxy_path.is_file():
        pytest.fail(f"HCU mooncake proxy missing at {proxy_path}")
    spec = importlib.util.spec_from_file_location(
        "mooncake_connector_proxy_hcu_under_test", proxy_path
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # optional deps (fastapi/httpx) absent
        pytest.skip(f"cannot import mooncake proxy: {exc!r}")
    for required in ("_stream_guarded_by_prefill", "_report_prefill_failure"):
        if not hasattr(module, required):
            pytest.fail(f"{proxy_path.name} is missing {required}()")
    return module


def test_proxy_aborts_decode_leg_when_prefill_fails() -> None:
    proxy = _load_proxy_module()

    closed = asyncio.Event()

    async def never_yields():
        # The decoder produces nothing until the producer's KV arrives.
        try:
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            closed.set()

    async def scenario():
        async def failing_prefill():
            raise RuntimeError("ReadError('')")

        prefill_task = asyncio.create_task(failing_prefill())
        chunks = []
        with pytest.raises(RuntimeError, match="prefill leg failed"):
            async for chunk in proxy._stream_guarded_by_prefill(
                prefill_task, never_yields(), "req-1"
            ):
                chunks.append(chunk)
        assert chunks == []
        # The decode stream is closed, so the decoder sees a disconnect and
        # aborts instead of holding its blocks for the KV abort timeout.
        await asyncio.wait_for(closed.wait(), timeout=5)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_proxy_relays_all_chunks_when_prefill_succeeds() -> None:
    proxy = _load_proxy_module()

    async def three_chunks():
        for payload in (b"a", b"b", b"c"):
            await asyncio.sleep(0)
            yield payload

    async def scenario():
        async def ok_prefill():
            await asyncio.sleep(0.05)

        prefill_task = asyncio.create_task(ok_prefill())
        chunks = [
            chunk
            async for chunk in proxy._stream_guarded_by_prefill(
                prefill_task, three_chunks(), "req-2"
            )
        ]
        assert chunks == [b"a", b"b", b"c"]
        # The decode stream may well finish first (full prefix-cache hit on D);
        # the guard must not wait on the prefill leg in that case.
        await prefill_task

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_proxy_logs_prefill_failure_that_loses_the_race(capsys) -> None:
    """A prefill that fails after the decode stream ended is still reported."""
    proxy = _load_proxy_module()

    async def scenario():
        async def late_failing_prefill():
            await asyncio.sleep(0.05)
            raise RuntimeError("ReadError('')")

        prefill_task = asyncio.create_task(late_failing_prefill())
        proxy._report_prefill_failure(prefill_task, "req-3")
        with pytest.raises(RuntimeError):
            await prefill_task

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))

    out = capsys.readouterr().out
    assert "Prefill leg failed for request req-3" in out
