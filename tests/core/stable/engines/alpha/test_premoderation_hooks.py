# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
from dataclasses import replace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from lagom import Container
import pytest

from parlant.core.agents import Agent, CompositionMode, MessageOutputMode
from parlant.core.customers import CustomerStore
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseGenerator,
    _CannedResponseSelectionResult,
)
from parlant.core.engines.alpha.engine_context import (
    EngineContext,
    Interaction,
    ResponseState,
    is_premoderation_required,
)
from parlant.core.engines.alpha.hooks import EngineHookResult, EngineHooks, ToolBatchExecution
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallInferenceResult,
    ToolCallId,
    ToolCallResult,
    ToolCaller,
    ToolInsights,
    ToolResultId,
)
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.types import Context
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.sessions import EventKind, EventSource, MessageEventData, SessionStore
from parlant.core.tools import ToolContext, ToolError, ToolId, ToolResult
from parlant.core.tracer import Tracer

from tests.core.common.utils import create_event_message


def _generation_info() -> GenerationInfo:
    return GenerationInfo(
        schema_name="test",
        model="test",
        duration=0.0,
        usage=UsageInfo(input_tokens=0, output_tokens=0),
    )


async def _engine_context(
    container: Container,
    agent: Agent,
    metadata: dict[str, Any] | None = None,
) -> tuple[EngineContext, EventBuffer]:
    customer = await container[CustomerStore].read_customer(CustomerStore.GUEST_ID)
    session = await container[SessionStore].create_session(customer.id, agent.id)
    event_buffer = EventBuffer(agent)
    context = EngineContext(
        info=Context(session_id=session.id, agent_id=agent.id),
        logger=container[Logger],
        tracer=container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=event_buffer,
        response_event_emitter=EventBuffer(agent),
        interaction=Interaction(
            events=[
                create_event_message(
                    offset=0,
                    source=EventSource.CUSTOMER,
                    message="Hello",
                    customer=customer,
                    metadata=metadata or {},
                )
            ]
        ),
        state=ResponseState(
            context_variables=[],
            glossary_terms=set(),
            capabilities=[],
            iterations=[],
            ordinary_guideline_matches=[],
            tool_enabled_guideline_matches={},
            journeys=[],
            journey_paths={},
            tool_events=[],
            tool_insights=ToolInsights(),
            prepared_to_respond=False,
            message_events=[],
        ),
    )
    return context, event_buffer


async def test_that_empty_message_batch_is_reported_once(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[MessageGenerator]
    hooks = container[EngineHooks]
    handler = AsyncMock(return_value=EngineHookResult.CALL_NEXT)
    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "shots", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generator,
        "_generate_response_message",
        AsyncMock(return_value=(_generation_info(), None)),
    )

    result = await generator.generate_response(context)

    handler.assert_awaited_once_with(context, [], None)
    assert result[0].events == []
    assert not [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]


async def test_that_fluid_message_is_normalized_before_emit(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[MessageGenerator]
    hooks = container[EngineHooks]
    received: list[list[MessageEventData]] = []

    async def handler(
        context: EngineContext,
        payload: list[MessageEventData],
        exc: Exception | None,
    ) -> EngineHookResult:
        assert not [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]
        received.append(payload)
        return EngineHookResult.BAIL

    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "shots", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generator,
        "_generate_response_message",
        AsyncMock(return_value=(_generation_info(), "Hello from FLUID")),
    )

    result = await generator.generate_response(context)

    assert received == [
        [
            {
                "message": "Hello from FLUID",
                "participant": {"id": agent.id, "display_name": agent.name},
            }
        ]
    ]
    assert result[0].events == []
    assert not [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]


async def test_that_canned_main_and_followups_are_held_as_one_batch(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[CannedResponseGenerator]
    hooks = container[EngineHooks]
    main = _CannedResponseSelectionResult(
        message="one\n\ntwo",
        draft=None,
        rendered_canned_responses=[],
        chosen_canned_responses=[],
    )
    follow_up = _CannedResponseSelectionResult(
        message="three",
        draft=None,
        rendered_canned_responses=[],
        chosen_canned_responses=[],
    )
    policy = Mock()
    policy.is_message_splitting_required = AsyncMock(return_value=True)
    policy.get_follow_up_delay = AsyncMock(return_value=0.0)
    received: list[list[MessageEventData]] = []

    async def handler(
        context: EngineContext,
        payload: list[MessageEventData],
        exc: Exception | None,
    ) -> EngineHookResult:
        assert event_buffer.events == []
        received.append(payload)
        return EngineHookResult.BAIL

    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "_get_relevant_canned_responses", AsyncMock(return_value=[]))
    monkeypatch.setattr(generator, "_generate_response", AsyncMock(return_value=({}, main)))
    monkeypatch.setattr(
        generator,
        "generate_follow_up_response",
        AsyncMock(return_value=({}, follow_up)),
    )
    monkeypatch.setattr(
        generator._perceived_performance_policy_provider,
        "get_policy",
        Mock(return_value=policy),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    result = await generator.generate_response(context)

    assert [[message["message"] for message in batch] for batch in received] == [
        ["one", "two", "three"]
    ]
    assert result[0].events == []
    assert event_buffer.events == []


async def test_that_premoderated_streaming_uses_buffered_generation(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = await _engine_context(
        container,
        agent,
        {"_impact_private": {"impact_cycle_v1": {"premoderation_required": True}}},
    )
    context.agent = replace(
        agent,
        composition_mode=CompositionMode.CANNED_FLUID,
        message_output_mode=MessageOutputMode.STREAM,
    )
    generator = container[CannedResponseGenerator]
    main = _CannedResponseSelectionResult(
        message="buffered",
        draft=None,
        rendered_canned_responses=[],
        chosen_canned_responses=[],
    )
    streaming = AsyncMock(return_value=[])
    buffered = AsyncMock(return_value=({}, main))
    policy = Mock()
    policy.is_message_splitting_required = AsyncMock(return_value=False)
    monkeypatch.setattr(generator, "_generate_streaming_response", streaming)
    monkeypatch.setattr(generator, "_get_relevant_canned_responses", AsyncMock(return_value=[]))
    monkeypatch.setattr(generator, "_generate_response", buffered)
    monkeypatch.setattr(
        generator._perceived_performance_policy_provider,
        "get_policy",
        Mock(return_value=policy),
    )
    generator.disable_follow_ups()

    result = await generator.generate_response(context)

    streaming.assert_not_awaited()
    buffered.assert_awaited_once()
    emitted_events = [event for event in result[0].events if event is not None]
    assert [cast(MessageEventData, event.data)["message"] for event in emitted_events] == [
        "buffered"
    ]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, False),
        ({"_impact_private": {"impact_cycle_v1": {"premoderation_required": False}}}, False),
        ({"_impact_private": {"impact_cycle_v1": {"premoderation_required": True}}}, True),
        ({"_impact_private": "malformed"}, True),
        ({"_impact_private": {"impact_cycle_v1": "malformed"}}, True),
        ({"_impact_private": {"impact_cycle_v1": {}}}, True),
    ],
)
async def test_that_malformed_private_cycle_fails_closed(
    container: Container,
    agent: Agent,
    metadata: dict[str, Any],
    expected: bool,
) -> None:
    context, _ = await _engine_context(container, agent, metadata)
    assert is_premoderation_required(context) is expected


async def test_that_direct_tool_message_is_blocked_during_premoderation() -> None:
    emitted: list[str] = []

    async def emit_message(message: str) -> None:
        emitted.append(message)

    context = ToolContext(
        agent_id="agent",
        session_id="session",
        customer_id="customer",
        emit_message=emit_message,
        premoderation_required=True,
    )

    with pytest.raises(ToolError):
        await context.emit_message("must not escape")

    assert emitted == []


async def test_that_premoderation_keeps_custom_tool_emission() -> None:
    emitted: list[Any] = []

    async def emit_custom(data: Any) -> None:
        emitted.append(data)

    context = ToolContext(
        agent_id="agent",
        session_id="session",
        customer_id="customer",
        emit_custom=emit_custom,
        premoderation_required=True,
    )

    await context.emit_custom({"kind": "progress"})

    assert emitted == [{"kind": "progress"}]


@pytest.mark.parametrize(
    ("hook_result", "expected_names", "expected_events", "expected_halted"),
    [
        (EngineHookResult.BAIL, [], 0, True),
        (EngineHookResult.RESOLVE, ["safe", "dangerous"], 2, False),
    ],
)
async def test_that_mixed_consequential_batch_is_held_before_execution(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    hook_result: EngineHookResult,
    expected_names: list[str],
    expected_events: int,
    expected_halted: bool,
) -> None:
    context, _ = await _engine_context(
        container,
        agent,
        {"_impact_private": {"impact_cycle_v1": {"premoderation_required": True}}},
    )
    calls = [
        ToolCall(ToolCallId("safe-call"), ToolId("local", "safe"), {}),
        ToolCall(ToolCallId("dangerous-call"), ToolId("local", "dangerous"), {}),
    ]
    generator = container[ToolEventGenerator]
    tool_caller = container[ToolCaller]
    hooks = container[EngineHooks]
    service = AsyncMock()
    service.resolve_tool.side_effect = lambda name, _: Mock(consequential=name == "dangerous")
    monkeypatch.setattr(
        generator._service_registry,
        "read_tool_service",
        AsyncMock(return_value=service),
    )
    executed: list[str] = []

    async def execute_calls(
        tool_context: ToolContext, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        executed.extend(call.tool_id.tool_name for call in tool_calls)
        return [
            ToolCallResult(
                ToolResultId(f"result-{call.id}"),
                call,
                cast(
                    Any,
                    {
                        "data": "ok",
                        "metadata": {},
                        "control": {},
                        "canned_responses": [],
                        "canned_response_fields": {},
                        "guidelines": [],
                    },
                ),
            )
            for call in tool_calls
        ]

    monkeypatch.setattr(tool_caller, "execute_tool_calls", execute_calls)
    approval = AsyncMock(return_value=hook_result)
    hooks.on_consequential_tool_batch_generated.append(approval)

    execution, halted = await generator.execute_tool_calls(context, calls)

    assert executed == expected_names
    assert len(execution.events) == expected_events
    assert halted is expected_halted
    approval.assert_awaited_once()
    assert approval.await_args is not None
    assert approval.await_args.args[1] == [calls[1]]


async def test_that_approved_mixed_batch_executes_as_one_native_batch(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = await _engine_context(
        container,
        agent,
        {"_impact_private": {"impact_cycle_v1": {"premoderation_required": True}}},
    )
    calls = [
        ToolCall(ToolCallId("safe-call"), ToolId("local", "safe"), {}),
        ToolCall(ToolCallId("dangerous-call"), ToolId("local", "dangerous"), {}),
    ]
    generator = container[ToolEventGenerator]
    service = AsyncMock()
    service.resolve_tool.side_effect = lambda name, _: Mock(consequential=name == "dangerous")
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: list[str] = []

    async def call_tool(
        name: str,
        tool_context: ToolContext,
        arguments: Any,
    ) -> ToolResult:
        started.append(name)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return ToolResult("ok")

    service.call_tool.side_effect = call_tool
    monkeypatch.setattr(
        generator._service_registry,
        "read_tool_service",
        AsyncMock(return_value=service),
    )
    container[EngineHooks].on_consequential_tool_batch_generated.append(
        AsyncMock(return_value=EngineHookResult.RESOLVE)
    )

    execution_task = asyncio.create_task(generator.execute_tool_calls(context, calls))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert sorted(started) == ["dangerous", "safe"]
    assert not execution_task.done()
    release.set()

    execution, halted = await execution_task

    assert halted is False
    assert execution.calls == calls
    assert len(execution.results) == 2


async def test_that_post_execution_hook_receives_calls_results_and_events(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = await _engine_context(container, agent)
    call = ToolCall(ToolCallId("call"), ToolId("local", "safe"), {})
    result = ToolCallResult(
        ToolResultId("result"),
        call,
        cast(
            Any,
            {
                "data": "ok",
                "metadata": {},
                "control": {},
                "canned_responses": [],
                "canned_response_fields": {},
                "guidelines": [],
            },
        ),
    )
    generator = container[ToolEventGenerator]
    monkeypatch.setattr(
        container[ToolCaller], "execute_tool_calls", AsyncMock(return_value=[result])
    )

    async def after_execution(
        hook_context: EngineContext,
        payload: ToolBatchExecution,
        exc: Exception | None,
    ) -> EngineHookResult:
        assert hook_context.state.tool_events == list(payload.events)
        return EngineHookResult.CALL_NEXT

    handler = AsyncMock(side_effect=after_execution)
    container[EngineHooks].on_tool_batch_executed.append(handler)

    execution, halted = await generator.execute_tool_calls(context, [call])

    assert halted is False
    assert execution.calls == [call]
    assert execution.results == [result]
    assert len(execution.events) == 1
    handler.assert_awaited_once()
    assert handler.await_args is not None
    payload = cast(ToolBatchExecution, handler.await_args.args[1])
    assert payload == execution


async def test_that_post_execution_bail_stages_results_and_stops_preparation(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = await _engine_context(container, agent)
    call = ToolCall(ToolCallId("call"), ToolId("local", "safe"), {})
    result = ToolCallResult(
        ToolResultId("result"),
        call,
        cast(
            Any,
            {
                "data": "done",
                "metadata": {},
                "control": {},
                "canned_responses": [],
                "canned_response_fields": {},
                "guidelines": [],
            },
        ),
    )
    generator = container[ToolEventGenerator]
    monkeypatch.setattr(
        container[ToolCaller], "execute_tool_calls", AsyncMock(return_value=[result])
    )
    container[EngineHooks].on_tool_batch_executed.append(
        AsyncMock(return_value=EngineHookResult.BAIL)
    )

    execution, halted = await generator.execute_tool_calls(context, [call])

    assert halted is True
    assert context.state.tool_events == list(execution.events)


async def test_that_approved_results_are_available_to_the_next_iteration(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = await _engine_context(container, agent)
    call = ToolCall(ToolCallId("call"), ToolId("local", "safe"), {})
    result = ToolCallResult(
        ToolResultId("result"),
        call,
        cast(
            Any,
            {
                "data": "done",
                "metadata": {},
                "control": {},
                "canned_responses": [],
                "canned_response_fields": {},
                "guidelines": [],
            },
        ),
    )
    generator = container[ToolEventGenerator]
    tool_caller = container[ToolCaller]
    monkeypatch.setattr(tool_caller, "execute_tool_calls", AsyncMock(return_value=[result]))
    execution, halted = await generator.execute_tool_calls(context, [call])
    assert halted is False

    context.state.tool_enabled_guideline_matches = {cast(Any, object()): [call.tool_id]}
    captured_staged_events: list[Any] = []

    async def infer(context: Any) -> ToolCallInferenceResult:
        captured_staged_events.extend(context.staged_events)
        return ToolCallInferenceResult(
            total_duration=0,
            batch_count=0,
            batch_generations=[],
            batches=[],
            insights=ToolInsights(),
        )

    monkeypatch.setattr(tool_caller, "infer_tool_calls", infer)
    preexecution_state = await generator.create_preexecution_state(
        context.session_event_emitter,
        context.session.id,
        context.agent,
        context.customer,
        [],
        context.interaction.events,
        [],
        [],
        context.state.tool_enabled_guideline_matches,
        context.state.tool_events,
    )

    await generator.infer_tool_calls(preexecution_state, context)

    assert captured_staged_events == list(execution.events)
