# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timezone

from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.sessions import Event, EventId, EventKind, EventSource


def test_that_inspector_events_are_excluded_from_interaction_history_but_custom_events_remain(
) -> None:
    def event(event_id: str, kind: EventKind, message: str, offset: int) -> Event:
        return Event(
            id=EventId(event_id),
            source=EventSource.CUSTOMER if kind == EventKind.MESSAGE else EventSource.AI_AGENT,
            kind=kind,
            creation_utc=datetime.now(timezone.utc),
            offset=offset,
            trace_id="trace-1",
            data={
                "message": message,
                "participant": {"display_name": "Customer"},
            },
            metadata={},
            deleted=False,
        )

    prompt = (
        PromptBuilder()
        .add_interaction_history(
            [
                event("customer-message", EventKind.MESSAGE, "Customer-visible message", 0),
                event("inspector-snapshot", EventKind.INSPECTOR, "Private inspector draft", 1),
                event("custom-event", EventKind.CUSTOM, "Intentional custom context", 2),
            ]
        )
        .build()
    )

    assert "Customer-visible message" in prompt
    assert "Private inspector draft" not in prompt
    assert "Intentional custom context" in prompt
