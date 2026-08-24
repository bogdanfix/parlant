# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Coroutine, cast
from unittest.mock import AsyncMock

from parlant.core.app_modules.sessions import SessionModule
from parlant.core.sessions import EventId, Session, SessionId


class BackgroundTasks:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task
        self.cancelled_tags: list[str] = []
        self.restarted_tags: list[str] = []

    async def cancel(self, *, tag: str, reason: str) -> None:
        self.cancelled_tags.append(tag)
        self.task.cancel(reason)

    async def restart(self, coroutine: Coroutine[object, object, None], *, tag: str) -> asyncio.Task[None]:
        self.restarted_tags.append(tag)
        self.task = asyncio.create_task(coroutine)
        return self.task


def module_with_task(
    session_id: SessionId,
    trigger_event_id: EventId,
    task: asyncio.Task[None],
) -> tuple[SessionModule, BackgroundTasks]:
    module = SessionModule.__new__(SessionModule)
    background_tasks = BackgroundTasks(task)
    module._session_store = cast(
        object,
        SimpleNamespace(read_session=AsyncMock(return_value=None)),
    )
    module._background_task_service = cast(object, background_tasks)
    module._processing_tasks = {session_id: (trigger_event_id, task)}
    module._processing_tasks_lock = asyncio.Lock()
    return module, background_tasks


async def test_that_matching_trigger_cancels_current_processing() -> None:
    session_id = SessionId("session-1")
    trigger_event_id = EventId("event-1")
    task = asyncio.create_task(asyncio.Event().wait())
    module, background_tasks = module_with_task(session_id, trigger_event_id, task)

    assert await module.cancel_processing(session_id, trigger_event_id) == "cancelled"
    await asyncio.sleep(0)

    assert task.cancelled()
    assert background_tasks.cancelled_tags == ["process-session(session-1)"]


async def test_that_old_trigger_does_not_cancel_new_processing() -> None:
    session_id = SessionId("session-1")
    task = asyncio.create_task(asyncio.Event().wait())
    module, background_tasks = module_with_task(session_id, EventId("event-2"), task)

    assert await module.cancel_processing(session_id, EventId("event-1")) == "not_current"
    assert not task.done()
    assert background_tasks.cancelled_tags == []

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def test_that_finished_processing_returns_already_finished() -> None:
    session_id = SessionId("session-1")
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    module, background_tasks = module_with_task(session_id, EventId("event-1"), task)

    assert await module.cancel_processing(session_id, EventId("event-1")) == "already_finished"
    assert background_tasks.cancelled_tags == []


async def test_that_next_inbound_dispatch_replaces_cancelled_processing() -> None:
    session_id = SessionId("session-1")
    old_task = asyncio.create_task(asyncio.Event().wait())
    module, background_tasks = module_with_task(session_id, EventId("event-1"), old_task)
    module._tracer = cast(object, SimpleNamespace(trace_id="trace-2"))
    module._process_session = AsyncMock(side_effect=lambda _: asyncio.Event().wait())

    assert await module.cancel_processing(session_id, EventId("event-1")) == "cancelled"
    await asyncio.sleep(0)
    await module.dispatch_processing_task(
        cast(Session, SimpleNamespace(id=session_id)),
        trigger_event_id=EventId("event-2"),
    )

    current_trigger, current_task = module._processing_tasks[session_id]
    assert current_trigger == EventId("event-2")
    assert not current_task.done()
    assert background_tasks.restarted_tags == ["process-session(session-1)"]

    current_task.cancel()
    with suppress(asyncio.CancelledError):
        await current_task


async def test_that_cancellation_does_not_replay_started_side_effect_or_emit_message() -> None:
    session_id = SessionId("session-1")
    side_effects: list[str] = []
    messages: list[str] = []
    started = asyncio.Event()

    async def processing() -> None:
        side_effects.append("executed")
        started.set()
        await asyncio.Event().wait()
        messages.append("late message")

    task = asyncio.create_task(processing())
    await started.wait()
    module, _ = module_with_task(session_id, EventId("event-1"), task)

    assert await module.cancel_processing(session_id, EventId("event-1")) == "cancelled"
    await asyncio.sleep(0)

    assert side_effects == ["executed"]
    assert messages == []
