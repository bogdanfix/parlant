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

import asyncio
import time

import pytest

from parlant.adapters.nlp.emcie_service import _KeyPool, _KeySlot
from parlant.core.nlp.policies import RateLimitPolicy


async def test_that_key_pool_parses_comma_separated_keys() -> None:
    pool = _KeyPool(keys=["key1", "key2", "key3"], rpm_limit=100, tpm_limit=0)

    assert pool.total_key_count == 3

    seen: set[int] = set()
    for _ in range(3):
        slot = await pool.acquire()
        seen.add(slot.index)

    assert seen == {1, 2, 3}


async def test_that_key_pool_cycles_keys_in_round_robin() -> None:
    pool = _KeyPool(keys=["key1", "key2", "key3"], rpm_limit=100, tpm_limit=0)

    first = await pool.acquire()
    second = await pool.acquire()
    third = await pool.acquire()
    fourth = await pool.acquire()

    assert first.index == 1
    assert second.index == 2
    assert third.index == 3
    assert fourth.index == 1


async def test_that_key_pool_skips_exhausted_keys(
) -> None:
    pool = _KeyPool(keys=["key1", "key2", "key3"], rpm_limit=100, tpm_limit=0)

    slot1 = await pool.acquire()
    assert slot1.index == 1

    pool.mark_exhausted(slot1)

    slot2 = await pool.acquire()
    assert slot2.index == 2

    slot3 = await pool.acquire()
    assert slot3.index == 3


async def test_that_key_pool_skips_dead_keys(
) -> None:
    pool = _KeyPool(keys=["key1", "key2", "key3"], rpm_limit=100, tpm_limit=0)

    slot1 = await pool.acquire()
    assert slot1.index == 1

    pool.mark_dead(slot1)

    slot2 = await pool.acquire()
    assert slot2.index == 2

    slot3 = await pool.acquire()
    assert slot3.index == 3

    # Dead key remains dead — skipped again
    slot4 = await pool.acquire()
    assert slot4.index == 2


async def test_that_key_pool_waits_when_all_exhausted(
) -> None:
    pool = _KeyPool(keys=["key1", "key2"], rpm_limit=100, tpm_limit=0)

    slot1 = await pool.acquire()
    slot2 = await pool.acquire()

    pool.mark_exhausted(slot1)
    pool.mark_exhausted(slot2)

    t0 = time.monotonic()
    slot3 = await pool.acquire()
    elapsed = time.monotonic() - t0

    assert elapsed >= 55.0  # 60s cooldown, minus some tolerance
    assert slot3.index in (1, 2)


async def test_that_key_pool_raises_when_all_keys_dead(
) -> None:
    from parlant.adapters.nlp.emcie_service import EmcieAPIError

    pool = _KeyPool(keys=["key1", "key2"], rpm_limit=100, tpm_limit=0)

    slot1 = await pool.acquire()
    slot2 = await pool.acquire()

    pool.mark_dead(slot1)
    pool.mark_dead(slot2)

    with pytest.raises(EmcieAPIError):
        await pool.acquire()


async def test_that_key_pool_recovers_exhausted_key_after_cooldown(
) -> None:
    pool = _KeyPool(keys=["key1"], rpm_limit=100, tpm_limit=0)

    slot1 = await pool.acquire()
    pool.mark_exhausted(slot1)

    # Ключ один — принудительно сдвигаем exhausted_until в прошлое
    slot1.exhausted_until = time.monotonic() - 1.0

    slot2 = await pool.acquire()
    assert slot2.index == 1


async def test_that_key_pool_returns_single_key_for_no_commas(
) -> None:
    pool = _KeyPool(keys=["single-key"], rpm_limit=100, tpm_limit=0)

    assert pool.total_key_count == 1

    for _ in range(5):
        slot = await pool.acquire()
        assert slot.index == 1
        assert slot.value == "single-key"


async def test_that_each_key_has_independent_rate_limiter(
) -> None:
    pool = _KeyPool(keys=["key1", "key2"], rpm_limit=1, tpm_limit=0)

    slot1 = await pool.acquire()
    assert slot1.index == 1
    assert slot1.limiter.max_requests == 1

    slot2 = await pool.acquire()
    assert slot2.index == 2
    assert slot2.limiter.max_requests == 1

    # У каждого ключа свой limiter — они не блокируют друг друга
    t0 = time.monotonic()
    await slot1.limiter.wait_and_record()
    await slot2.limiter.wait_and_record()
    elapsed = time.monotonic() - t0

    # Оба лимитера должны позволить запрос без ожидания
    assert elapsed < 0.1


async def test_that_each_key_has_independent_token_budget(
) -> None:
    pool = _KeyPool(keys=["key1", "key2"], rpm_limit=100, tpm_limit=100)

    slot1 = await pool.acquire()
    slot2 = await pool.acquire()

    # Заполняем бюджет ключа 1, но не ключа 2
    await slot1.budget.report(100)

    t0 = time.monotonic()
    await slot1.budget.acquire()
    elapsed1 = time.monotonic() - t0

    t0 = time.monotonic()
    await slot2.budget.acquire()
    elapsed2 = time.monotonic() - t0

    # Ключ 1 ждёт (бюджет заполнен), ключ 2 не ждёт
    assert elapsed1 >= 0.1
    assert elapsed2 < 0.1


async def test_that_key_mask_hides_full_key(
) -> None:
    slot = _KeySlot(index=1, value="sk-very-long-api-key-12345678", rpm_limit=100, tpm_limit=0)
    mask = slot.mask()
    assert "sk-very-long-api-key-12345678" not in mask
    assert "ключ #1" in mask
    assert "sk-v" in mask
    assert "5678" in mask


async def test_that_key_mask_handles_short_key(
) -> None:
    slot = _KeySlot(index=5, value="short", rpm_limit=100, tpm_limit=0)
    mask = slot.mask()
    assert "short" not in mask
    assert "ключ #5" in mask
    assert "***" in mask


async def test_that_wait_and_record_returns_sleep_time(
) -> None:
    limiter = RateLimitPolicy(max_requests=1, per_seconds=0.2)

    await limiter.wait_and_record()

    t0 = time.monotonic()
    sleep_time = await limiter.wait_and_record()
    elapsed = time.monotonic() - t0

    assert sleep_time >= 0.15
    assert elapsed >= 0.15


async def test_that_wait_and_record_returns_zero_when_no_limit(
) -> None:
    limiter = RateLimitPolicy(max_requests=0)

    sleep_time = await limiter.wait_and_record()
    assert sleep_time == 0.0
