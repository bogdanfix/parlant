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

from lagom import Container

from parlant.core.emissions import EventEmitterFactory
from parlant.core.sessions import EventKind, Session, SessionStore


async def test_that_event_publisher_emits_and_persists_dedicated_inspector_events(
    container: Container,
    new_session: Session,
) -> None:
    store = container[SessionStore]
    emitter = await container[EventEmitterFactory].create_event_emitter(
        new_session.agent_id, new_session.id
    )

    event = await emitter.emit_inspector_event(
        trace_id="trace-1",
        data={"schema": "builder.dialog-inspector.v1", "phase": "generated"},
    )

    assert event.kind == EventKind.INSPECTOR
    assert (await store.list_events(new_session.id))[-1].kind == EventKind.INSPECTOR
