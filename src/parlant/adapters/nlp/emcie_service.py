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

from __future__ import annotations
import asyncio
from pprint import pformat
import re
import time
from typing import Any, AsyncIterator, Callable, Mapping, TypeAlias, cast
from httpx import AsyncClient
import httpx
from typing_extensions import Literal, override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.policies import RateLimitPolicy
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    EmbedderHints,
    ModelSize,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    BaseStreamingTextGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)
from parlant.core.tracer import Tracer
from parlant.core.version import VERSION


RATE_LIMIT_ERROR_MESSAGE = (
    "Emcie API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Emcie account balance and billing status.\n"
    "- Review your API usage limits in Emcie's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://docs.emcie.co\n"
)

GenerationModelTier: TypeAlias = Literal["jackal", "bison"]
EmbeddingModelTier: TypeAlias = Literal["jackal-embedding", "bison-embedding"]
ModelRole: TypeAlias = Literal["teacher", "student", "auto"]

BASE_URL = os.environ.get("EMCIE_API_URL", "https://api.emcie.co/inference")

# Pattern to detect word boundaries for chunking
# Matches after any whitespace character
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")

# Number of words to buffer before yielding a chunk
_WORDS_PER_CHUNK = 3



class _TokenBudget:
    def __init__(self, max_tokens_per_minute: int, window_seconds: float = 60.0) -> None:
        self._max_tokens_per_minute = max_tokens_per_minute
        self._window_seconds = window_seconds
        self._usages: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._max_tokens_per_minute <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            window_start = now - self._window_seconds
            self._usages = [(ts, t) for ts, t in self._usages if ts > window_start]
            total = sum(t for _, t in self._usages)

            while total >= self._max_tokens_per_minute and self._usages:
                sleep_time = self._usages[0][0] + self._window_seconds - now
                if sleep_time <= 0:
                    break
                await asyncio.sleep(sleep_time)
                now = time.monotonic()
                window_start = now - self._window_seconds
                self._usages = [(ts, t) for ts, t in self._usages if ts > window_start]
                total = sum(t for _, t in self._usages)

    async def report(self, tokens: int) -> None:
        if self._max_tokens_per_minute <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            window_start = now - self._window_seconds
            self._usages = [(ts, t) for ts, t in self._usages if ts > window_start]
            self._usages.append((now, tokens))


class _KeySlot:
    def __init__(self, index: int, value: str, rpm_limit: int, tpm_limit: int) -> None:
        self.index = index
        self.value = value
        self.limiter = RateLimitPolicy(max_requests=rpm_limit)
        self.budget = _TokenBudget(max_tokens_per_minute=tpm_limit)
        self.exhausted_until: float = 0.0
        self.dead: bool = False

    def is_available(self) -> bool:
        return not self.dead and time.monotonic() >= self.exhausted_until

    def mask(self) -> str:
        if len(self.value) <= 8:
            return f"ключ #{self.index} (***)"
        return f"ключ #{self.index} ({self.value[:4]}...{self.value[-4:]})"


class _KeyPool:
    _LOCK_COOLDOWN_SECONDS = 60.0

    def __init__(self, keys: list[str], rpm_limit: int, tpm_limit: int) -> None:
        self._slots = [
            _KeySlot(index=i + 1, value=k, rpm_limit=rpm_limit, tpm_limit=tpm_limit)
            for i, k in enumerate(keys)
        ]
        self._next_index = -1
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "_KeyPool":
        raw = os.environ.get("EMCIE_API_KEY", "")
        rpm = int(os.environ.get("EMCIE_RPM_LIMIT", "500"))
        tpm = int(os.environ.get("EMCIE_TPM_LIMIT", "0"))
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            keys = [""]
        return cls(keys=keys, rpm_limit=rpm, tpm_limit=tpm)

    @property
    def total_key_count(self) -> int:
        return len(self._slots)

    async def acquire(self) -> _KeySlot:
        while True:
            async with self._lock:
                for _ in range(len(self._slots)):
                    self._next_index = (self._next_index + 1) % len(self._slots)
                    slot = self._slots[self._next_index]
                    if slot.is_available():
                        return slot

                recoverable = [s for s in self._slots if not s.dead]
                if not recoverable:
                    raise EmcieAPIError(
                        "Все ключи Emcie недействительны. "
                        "Проверьте баланс и статус каждого аккаунта."
                    )

                earliest = min(recoverable, key=lambda s: s.exhausted_until)
                wait = earliest.exhausted_until - time.monotonic()

            if wait > 0:
                await asyncio.sleep(wait)

    def mark_exhausted(self, slot: _KeySlot) -> None:
        slot.exhausted_until = time.monotonic() + self._LOCK_COOLDOWN_SECONDS

    def mark_dead(self, slot: _KeySlot) -> None:
        slot.dead = True

    def has_dead_keys(self) -> bool:
        return any(s.dead for s in self._slots)

    def has_exhausted_keys(self) -> bool:
        return any(s.exhausted_until > time.monotonic() for s in self._slots if not s.dead)


_key_pool = _KeyPool.from_env()


class EmcieEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4.1")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class EmcieAPIError(Exception):
    pass


class InsufficientCreditsError(EmcieAPIError):
    pass


class RateLimitError(EmcieAPIError):
    pass


class UnauthorizedError(EmcieAPIError):
    pass


def _get_error_detail(response: httpx.Response) -> tuple[str, str]:
    try:
        error_message = (
            response.json().get("detail", {}).get("error", {}).get("message", "Unknown error")
        )
        request_id = response.json().get("detail", {}).get("request_id", "N/A")
    except Exception:
        try:
            error_message = response.text
        except Exception:
            error_message = "Unknown error (failed to parse error message)"

        request_id = "N/A"

    return error_message, request_id


class EmcieSchematicGenerator(BaseSchematicGenerator[T]):
    supported_emcie_params = ["temperature"]

    def __init__(
        self,
        model_name: str,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._model_role = model_role
        self._tokenizer = EmcieEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"emcie/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EmcieEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Emcie LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            props = prompt.props
            prompt = prompt.build()
        else:
            props = {}

        max_attempts = _key_pool.total_key_count * 3

        for _attempt in range(max_attempts):
            slot = await _key_pool.acquire()

            await slot.limiter.wait_and_record()
            await slot.budget.acquire()

            self.logger.info(f"Запрос через {slot.mask()}")

            try:
                t_start = time.time()

                timeout = httpx.Timeout(
                    connect=30.0,
                    read=120.0,
                    write=30.0,
                    pool=5.0,
                )

                async with AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{BASE_URL}/v1/completions",
                        headers={
                            "Authorization": f"Bearer {slot.value}",
                            "X-Parlant-Version": VERSION,
                        },
                        json={
                            "model_tier": self.model_name,
                            "model_role": self._model_role,
                            "prompt": prompt,
                            "schema_name": self.schema.__name__,
                            "hints": {
                                k: v for k, v in hints.items() if k in self.supported_emcie_params
                            },
                            "payload": props,
                        },
                    )

                    if response.status_code == 429:
                        _key_pool.mark_exhausted(slot)
                        self.logger.warning(f"{slot.mask()} исчерпан (429), переключаем...")
                        continue
                    elif response.status_code == 401:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (401 — неавторизован), переключаем..."
                        )
                        continue
                    elif response.status_code == 402:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (402 — недостаточно кредитов), переключаем..."
                        )
                        continue
                    elif response.status_code == 403:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (403 — неавторизован), переключаем..."
                        )
                        continue
                    elif response.is_error:
                        error_message, request_id = _get_error_detail(response)
                        if response.status_code >= 500:
                            self.logger.warning(
                                f"Ошибка сервера Emcie (status={response.status_code}) "
                                f"на {slot.mask()}, переключаем..."
                            )
                            await asyncio.sleep(1.0)
                            continue
                        raise EmcieAPIError(
                            f"Emcie API error: {response.status_code} {error_message} (RID={request_id})"
                        )

                    response.raise_for_status()

                t_end = time.time()
                break

            except EmcieAPIError:
                raise

        else:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise RateLimitError(
                "Все ключи Emcie исчерпаны. Дождитесь восстановления лимитов или "
                "проверьте баланс аккаунтов."
            )

        response_data = response.json()

        usage = response_data["usage"]
        cost = response_data["cost"]

        self.logger.trace(f"Emcie usage data:\n{pformat({**usage, **cost})}")

        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        total_tokens = input_tokens + output_tokens
        self.logger.info(
            f"Emcie ({slot.mask()}): {input_tokens} input + {output_tokens} output = {total_tokens} tokens"
        )

        await slot.budget.report(total_tokens)

        raw_content = response_data["completion"]

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        extra={},
                    ),
                ),
            )

        except ValidationError as e:
            self.logger.error(
                f"Error: {e.json(indent=2)}\nJSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Jackal(EmcieSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="jackal",
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Bison(EmcieSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="bison",
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


# ============================================================================
# Streaming Text Generators
# ============================================================================


class EmcieStreamingTextGenerator(BaseStreamingTextGenerator):
    """Streaming text generator using Emcie's streaming API.

    Buffers tokens into word-sized chunks for smoother frontend rendering.
    """

    supported_emcie_params = ["temperature"]

    def __init__(
        self,
        model_name: str,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)
        self._model_role = model_role
        self._tokenizer = EmcieEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"emcie-streaming/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EmcieEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        usage_info: UsageInfo | None = None

        async def chunk_generator() -> AsyncIterator[str | None]:
            nonlocal usage_info

            max_attempts = _key_pool.total_key_count * 3

            for _attempt in range(max_attempts):
                slot = await _key_pool.acquire()

                await slot.limiter.wait_and_record()
                await slot.budget.acquire()

                self.logger.info(f"Запрос (stream) через {slot.mask()}")

                timeout = httpx.Timeout(
                    connect=30.0,
                    read=120.0,
                    write=30.0,
                    pool=5.0,
                )

                buffer = ""

                try:
                    async with AsyncClient(timeout=timeout) as client:
                        async with client.stream(
                            "POST",
                            f"{BASE_URL}/v1/completions",
                            headers={
                                "Authorization": f"Bearer {slot.value}",
                                "X-Parlant-Version": VERSION,
                            },
                            json={
                                "model_tier": self.model_name,
                                "model_role": self._model_role,
                                "prompt": prompt,
                                "stream": True,
                                "hints": {
                                    k: v
                                    for k, v in hints.items()
                                    if k in self.supported_emcie_params
                                },
                            },
                        ) as response:
                            if response.is_error:
                                await response.aread()
                                error_message, request_id = _get_error_detail(response)

                            if response.status_code == 429:
                                _key_pool.mark_exhausted(slot)
                                self.logger.warning(
                                    f"{slot.mask()} исчерпан (429), переключаем..."
                                )
                                continue
                            elif response.status_code == 401:
                                _key_pool.mark_dead(slot)
                                self.logger.error(
                                    f"{slot.mask()} заблокирован (401 — неавторизован), переключаем..."
                                )
                                continue
                            elif response.status_code == 402:
                                _key_pool.mark_dead(slot)
                                self.logger.error(
                                    f"{slot.mask()} заблокирован (402 — недостаточно кредитов), переключаем..."
                                )
                                continue
                            elif response.status_code == 403:
                                _key_pool.mark_dead(slot)
                                self.logger.error(
                                    f"{slot.mask()} заблокирован (403 — неавторизован), переключаем..."
                                )
                                continue
                            elif response.status_code >= 500:
                                self.logger.warning(
                                    f"Ошибка сервера Emcie (status={response.status_code}) "
                                    f"на {slot.mask()}, переключаем..."
                                )
                                await asyncio.sleep(1.0)
                                continue

                            response.raise_for_status()

                            event_type: str | None = None

                            async for line in response.aiter_lines():
                                if line.startswith("event: "):
                                    event_type = line[7:]
                                elif line.startswith("data: ") and event_type:
                                    data = json.loads(line[6:])

                                    if event_type == "chunk":
                                        text = data.get("text", "")
                                        if text:
                                            buffer += text

                                            boundaries = list(
                                                _WORD_BOUNDARY_PATTERN.finditer(buffer)
                                            )
                                            if len(boundaries) >= _WORDS_PER_CHUNK:
                                                last_boundary = boundaries[_WORDS_PER_CHUNK - 1]
                                                chunk_text = buffer[: last_boundary.end()]
                                                buffer = buffer[last_boundary.end() :]
                                                yield chunk_text

                                    elif event_type == "done":
                                        usage = data.get("usage", {})
                                        usage_info = UsageInfo(
                                            input_tokens=int(usage.get("input_tokens", 0)),
                                            output_tokens=int(usage.get("output_tokens", 0)),
                                            extra={},
                                        )

                                        self.logger.trace(
                                            f"Emcie streaming usage data:\n{pformat(data)}"
                                        )

                                        input_tokens = int(usage.get("input_tokens", 0))
                                        output_tokens = int(usage.get("output_tokens", 0))
                                        self.logger.info(
                                            f"Emcie ({slot.mask()}): {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} tokens"
                                        )

                                        await slot.budget.report(input_tokens + output_tokens)

                                        if buffer:
                                            yield buffer
                                            buffer = ""

                                    elif event_type == "error":
                                        error_msg = data.get("error", {}).get(
                                            "message", "Unknown error"
                                        )
                                        raise EmcieAPIError(
                                            f"Emcie streaming error: {error_msg}"
                                        )

                except (httpx.TimeoutException, httpx.TransportError) as e:
                    self.logger.warning(
                        f"Сетевая ошибка на {slot.mask()}: {e}, переключаем..."
                    )
                    continue

                break

            else:
                self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
                raise RateLimitError(
                    "Все ключи Emcie исчерпаны при стриминговом запросе."
                )

            if usage_info is not None:
                await record_llm_metrics(
                    self.meter,
                    self.model_name,
                    schema_name="streaming",
                    input_tokens=usage_info.input_tokens,
                    output_tokens=usage_info.output_tokens,
                    cached_input_tokens=0,
                )

            yield None

        def get_usage() -> UsageInfo:
            if usage_info is None:
                return UsageInfo(input_tokens=0, output_tokens=0, extra={})
            return usage_info

        return chunk_generator(), get_usage


class JackalStreaming(EmcieStreamingTextGenerator):
    def __init__(
        self,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(
            model_name="jackal",
            model_role=model_role,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class BisonStreaming(EmcieStreamingTextGenerator):
    def __init__(
        self,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(
            model_name="bison",
            model_role=model_role,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


# ============================================================================
# Embedders
# ============================================================================


class EmcieEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name)
        self._tokenizer = EmcieEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"emcie/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EmcieEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        max_attempts = _key_pool.total_key_count * 3

        for _attempt in range(max_attempts):
            slot = await _key_pool.acquire()

            await slot.limiter.wait_and_record()
            await slot.budget.acquire()

            self.logger.info(f"Запрос (embed) через {slot.mask()}")

            try:
                timeout = httpx.Timeout(
                    connect=5.0,
                    read=120.0,
                    write=30.0,
                    pool=5.0,
                )

                async with AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{BASE_URL}/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {slot.value}",
                            "X-Parlant-Version": VERSION,
                        },
                        json={
                            "model_tier": self.model_name,
                            "inputs": texts,
                            "hints": {
                                k: v for k, v in hints.items() if k in self.supported_arguments
                            },
                        },
                    )

                    if response.is_error:
                        error_message, request_id = _get_error_detail(response)

                    if response.status_code == 429:
                        _key_pool.mark_exhausted(slot)
                        self.logger.warning(
                            f"{slot.mask()} исчерпан (429), переключаем..."
                        )
                        continue
                    elif response.status_code == 401:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (401 — неавторизован), переключаем..."
                        )
                        continue
                    elif response.status_code == 402:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (402 — недостаточно кредитов), переключаем..."
                        )
                        continue
                    elif response.status_code == 403:
                        _key_pool.mark_dead(slot)
                        self.logger.error(
                            f"{slot.mask()} заблокирован (403 — неавторизован), переключаем..."
                        )
                        continue
                    elif response.status_code >= 500:
                        self.logger.warning(
                            f"Ошибка сервера Emcie (status={response.status_code}) "
                            f"на {slot.mask()}, переключаем..."
                        )
                        await asyncio.sleep(1.0)
                        continue

                    response.raise_for_status()

            except EmcieAPIError as e:
                self.logger.error(f"Emcie API error occurred: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during Emcie API call: {e}")
                raise

            response_data = response.json()
            estimated_tokens = sum(len(t) // 4 for t in texts)
            await slot.budget.report(estimated_tokens)
            self.logger.info(
                f"Emcie ({slot.mask()}): ~{estimated_tokens} tokens estimated (embed)"
            )
            vectors = [data_point["embedding"] for data_point in response_data["data"]]
            return EmbeddingResult(vectors=vectors)

        self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
        raise RateLimitError("Все ключи Emcie исчерпаны при запросе эмбеддингов.")


class BisonEmbedding(EmcieEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="bison-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class JackalEmbedding(EmcieEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="jackal-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1536


class EmcieService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("EMCIE_API_KEY"):
            return """\
You're using Emcie's optimized NLP service, but EMCIE_API_KEY is not set.
Please set EMCIE_API_KEY in your environment before running Parlant.

For alternative providers, see https://parlant.io/docs/quickstart/installation.

Get an API key for Emcie by signing up at https://www.emcie.co."""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_tier: GenerationModelTier | None = None,
        model_role: ModelRole | None = None,
    ) -> None:
        self._logger = logger
        self._meter = meter
        self._tracer = tracer

        self._model_tier = model_tier or os.environ.get("EMCIE_MODEL_TIER", "jackal")
        self._model_role = model_role or os.environ.get("EMCIE_MODEL_ROLE", "auto")

        assert self._model_tier in ("jackal", "bison"), "Invalid EMCIE_MODEL_TIER"
        assert self._model_role in ("teacher", "student", "auto"), "Invalid EMCIE_MODEL_ROLE"

        self._logger.info("Initialized EmcieService")

    @property
    @override
    def supports_streaming(self) -> bool:
        return True

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        match self._model_tier:
            case "bison":
                return BisonStreaming(
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )
            case _:
                return JackalStreaming(
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> EmcieSchematicGenerator[T]:
        match self._model_tier:
            case "jackal":
                return Jackal[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )
            case "bison":
                return Bison[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )
            case _:
                raise ValueError(f"Unsupported model tier: {self._model_tier}")

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        match hints.get("model_size", ModelSize.AUTO):
            case ModelSize.AUTO | ModelSize.LARGE:
                return BisonEmbedding(self._logger, self._tracer, self._meter)
            case _:
                return JackalEmbedding(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
