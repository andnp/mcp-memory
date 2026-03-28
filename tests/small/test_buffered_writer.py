from __future__ import annotations

import threading

import pytest

from mcp_memory.storage.buffered_writer import BufferedWriter


pytestmark = pytest.mark.small


def test_buffered_writer_flushes_pending_items_on_close() -> None:
    written: list[list[int]] = []
    writer = BufferedWriter[int](
        lambda batch: written.append(list(batch)),
        name="test-close",
        low_watermark=10,
        high_watermark=10,
        flush_interval_seconds=60.0,
    )

    writer.write(1)
    writer.write(2)
    writer.close()

    assert written == [[1, 2]]


def test_buffered_writer_flushes_when_requested() -> None:
    written: list[list[int]] = []
    writer = BufferedWriter[int](
        lambda batch: written.append(list(batch)),
        name="test-flush",
        low_watermark=10,
        high_watermark=10,
        flush_interval_seconds=60.0,
    )

    try:
        writer.write_many([1, 2, 3])
        writer.flush()
    finally:
        writer.close()

    assert written == [[1, 2, 3]]


def test_buffered_writer_auto_flushes_at_low_watermark() -> None:
    written: list[list[int]] = []
    flushed = threading.Event()

    def write_batch(batch: list[int]) -> None:
        written.append(list(batch))
        flushed.set()

    writer = BufferedWriter[int](
        write_batch,
        name="test-watermark",
        low_watermark=1,
        high_watermark=10,
        flush_interval_seconds=0.01,
    )

    try:
        writer.write(7)
        assert flushed.wait(timeout=1.0) is True
    finally:
        writer.close()

    assert written == [[7]]
