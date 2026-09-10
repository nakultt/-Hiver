"""One thin OpenAI-compatible chat client.

Everything in ReplyBench that touches a model goes through here, so that:
  * every call is content-addressed and cached on disk (reruns are free and
    byte-identical, which is what makes an *evaluation* harness trustworthy);
  * we can swap Gemini / OpenAI / a local server by changing one base URL;
  * a deterministic "mock" backend lets the whole pipeline and the test suite
    run with no API key at all.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import CACHE_DIR, SETTINGS, Settings

T = TypeVar("T", bound=BaseModel)


_RETRY_HINT = re.compile(r"retry in ([\d.]+)")


def _retry_after(text: str) -> float:
    m = _RETRY_HINT.search(text or "")
    try:
        return float(m.group(1)) if m else 0.0
    except (TypeError, ValueError):
        return 0.0


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    """Raised instead of quietly burning a day's free-tier quota."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, pt: int, ct: int) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "by_model": self.by_model,
        }


def strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown fence if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def extract_json(text: str) -> Any:
    """Parse a model response that is *supposed* to be JSON.

    Even in JSON mode a model occasionally wraps output in prose or fences, so
    we degrade gracefully rather than throwing away a whole (paid) call.
    """
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("response was not JSON: " + repr(text[:400]))


class LLM:
    """Async, cached, retrying client for an OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or SETTINGS
        self.usage = Usage()
        self._sem = asyncio.Semaphore(self.s.concurrency)
        self._client: httpx.AsyncClient | None = None
        self._spent: dict[str, int] = {}
        # Free tiers rate-limit per MINUTE. Retrying into a 429 storm wastes both
        # wall clock and quota, so we pace request *starts* instead: one every
        # `_min_interval` seconds, globally. Far faster in practice than backoff.
        self._rpm = float(os.getenv("LLM_RPM", "18") or "18")
        self._min_interval = 60.0 / self._rpm if self._rpm > 0 else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def _pace(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            now = asyncio.get_event_loop().time()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait:
            await asyncio.sleep(wait)

    def _charge(self, model: str) -> None:
        """Count a live request and refuse to blow through the daily cap.

        Cache hits do not count -- only requests that actually leave the box.
        """
        cap = self.s.budget_for(model)
        spent = self._spent.get(model, 0)
        if cap and spent >= cap:
            raise BudgetExceeded(
                "daily request budget for " + model + " exhausted (" + str(cap) + "). "
                "Raise " + "BUDGET_" + model.upper().replace("-", "_").replace(".", "_")
                + " in .env, or wait for the quota to reset."
            )
        self._spent[model] = spent + 1

    @property
    def spent(self) -> dict[str, int]:
        return dict(self._spent)

    async def __aenter__(self) -> "LLM":
        self._client = httpx.AsyncClient(
            base_url=self.s.base_url,
            timeout=httpx.Timeout(240.0, connect=20.0),
            headers={"Authorization": "Bearer " + self.s.api_key},
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]
        model = str(payload.get("model", "x")).replace("/", "_")
        return CACHE_DIR / (model + "-" + digest + ".json")

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        variant: int = 0,
    ) -> str:
        """Return the assistant message content for `messages`.

        `variant` participates in the cache key but not the request: it is how
        we ask for k *independent* samples (used by the judge self-consistency
        check) without them collapsing onto a single cache entry.
        """
        model = model or self.s.gen_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        cache_key = dict(payload, __variant=variant, __backend=self.s.backend)
        path = self._cache_path(cache_key)
        if self.s.cache and path.exists():
            self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["content"]

        if not self.s.live:
            content = _mock_response(messages, json_mode=json_mode, variant=variant)
        else:
            self._charge(model)
            content = await self._post(payload, model)

        if self.s.cache:
            path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        return content

    async def _post(self, payload: dict[str, Any], model: str) -> str:
        assert self._client is not None, "use: async with LLM() as llm"
        delay = 2.0
        last: Exception | None = None
        for attempt in range(10):
            try:
                await self._pace()
                async with self._sem:
                    r = await self._client.post("chat/completions", json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    hint = _retry_after(r.text)
                    if hint:
                        await asyncio.sleep(min(hint + 1.0, 65.0))
                    raise LLMError("HTTP " + str(r.status_code) + ": " + r.text[:200])
                r.raise_for_status()
                body = r.json()
                choice = body["choices"][0]["message"]
                content = choice.get("content") or ""
                usage = body.get("usage") or {}
                self.usage.add(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                if not content.strip():
                    raise LLMError("empty completion")
                return content
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last = exc
                if attempt == 9:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 1.6, 45.0)
        raise LLMError("call failed after retries: " + str(last))

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed a batch of texts. Cached per-text so partial reuse works.

        Cached individually rather than per-batch: retrieval re-embeds the same
        corpus on every run but a different query each time, and a batch-level
        cache key would miss every time the query changed.
        """
        model = model or self.s.embed_model
        out: list[list[float]] = [[] for _ in texts]
        todo: list[int] = []
        for i, t in enumerate(texts):
            path = self._cache_path({"model": model, "embed": t, "__backend": self.s.backend})
            if self.s.cache and path.exists():
                self.usage.cache_hits += 1
                out[i] = json.loads(path.read_text(encoding="utf-8"))["v"]
            else:
                todo.append(i)

        for start in range(0, len(todo), 32):
            chunk = todo[start : start + 32]
            payload = {"model": model, "input": [texts[i] for i in chunk]}
            if not self.s.live:
                vecs = [_mock_embedding(texts[i]) for i in chunk]
            else:
                self._charge(model)
                vecs = await self._post_embed(payload)
            for i, v in zip(chunk, vecs):
                out[i] = v
                if self.s.cache:
                    self._cache_path({"model": model, "embed": texts[i],
                                      "__backend": self.s.backend}).write_text(
                        json.dumps({"v": v}), encoding="utf-8")
        return out

    async def _post_embed(self, payload: dict[str, Any]) -> list[list[float]]:
        assert self._client is not None, "use: async with LLM() as llm"
        delay = 2.0
        last: Exception | None = None
        for attempt in range(8):
            try:
                await self._pace()
                async with self._sem:
                    r = await self._client.post("embeddings", json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    hint = _retry_after(r.text)
                    if hint:
                        await asyncio.sleep(min(hint + 1.0, 65.0))
                    raise LLMError("HTTP " + str(r.status_code) + ": " + r.text[:160])
                r.raise_for_status()
                return [d["embedding"] for d in r.json()["data"]]
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt == 7:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 1.6, 45.0)
        raise LLMError("embed failed after retries: " + str(last))

    async def structured(
        self,
        messages: Sequence[dict[str, str]],
        schema: Type[T],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        variant: int = 0,
        repair_attempts: int = 2,
    ) -> T:
        """Call the model and validate the reply against a pydantic schema.

        On a validation failure we hand the model its own broken output plus the
        validator's complaint and ask for a fix. Two repair rounds is empirically
        enough; beyond that the prompt is the problem, not the sample.
        """
        convo = list(messages)
        last_err = ""
        for attempt in range(repair_attempts + 1):
            raw = await self.chat(
                convo,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                variant=variant + 1000 * attempt,
            )
            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, LLMError) as exc:
                last_err = str(exc)[:900]
                convo = list(messages) + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That did not validate against the required schema.\n"
                            "Validator said:\n" + last_err + "\n\n"
                            "Return ONLY corrected JSON matching the schema. No prose, no fences."
                        ),
                    },
                ]
        raise LLMError("schema " + schema.__name__ + " never validated: " + last_err)


async def gather_capped(coros: Sequence[Any], *, desc: str = "", quiet: bool = False) -> list[Any]:
    """asyncio.gather with a one-line progress ticker."""
    total = len(coros)
    if total == 0:
        return []
    done = 0
    started = time.time()
    results: list[Any] = [None] * total
    step = max(1, total // 10)

    async def run(i: int, c: Any) -> None:
        nonlocal done
        results[i] = await c
        done += 1
        if not quiet and (done % step == 0 or done == total):
            print("  " + desc + ": " + str(done) + "/" + str(total)
                  + "  (" + format(time.time() - started, ".1f") + "s)", flush=True)

    await asyncio.gather(*(run(i, c) for i, c in enumerate(coros)))
    return results


def _mock_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words vector for offline runs.

    Not semantic, but it is stable and gives similar text similar vectors, which
    is enough for the plumbing and the tests to exercise the dense path.
    """
    import math
    import re as _re

    v = [0.0] * dim
    for w in _re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _mock_response(messages: Sequence[dict[str, str]], *, json_mode: bool, variant: int) -> str:
    """Deterministic stand-in so the pipeline and tests run with no API key.

    Mock runs exercise plumbing only -- the numbers they produce are
    meaningless by construction, and the CLI says so loudly.
    """
    from .mockdata import mock_for

    joined = "\n".join(m.get("content", "") for m in messages)
    return mock_for(joined, json_mode=json_mode, variant=variant)
