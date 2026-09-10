"""End-to-end: generate replies for the eval set, then score every one."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

from ..config import RUNS_DIR
from ..llm import LLM, gather_capped
from ..schemas import AuditResult, Example, GenerationRecord, JudgeVerdict, ResponseScore
from ..generate.generator import SYSTEMS, generate_one
from ..generate.retrieve import Retriever
from .audit import audit as run_audit
from .judge import judge as run_judge
from .score import build_score


def eval_set(examples: Sequence[Example]) -> list[Example]:
    """What gets scored: the held-out test split plus the hand-authored hard cases."""
    return [e for e in examples if e.split in ("test", "hard")]


def corpus(examples: Sequence[Example]) -> list[Example]:
    """What the retriever may see. Train only -- never test, never hard."""
    return [e for e in examples if e.split == "train"]


async def generate_all(
    llm: LLM,
    examples: Sequence[Example],
    systems: Sequence[str] = SYSTEMS,
    *,
    limit: int | None = None,
) -> list[GenerationRecord]:
    retriever = Retriever(corpus(examples))
    # Embed the exemplar corpus once up front; cached on disk thereafter.
    await retriever.build_dense(llm)
    targets = eval_set(examples)[:limit] if limit else eval_set(examples)
    by_id = {e.id: e for e in targets}

    jobs = []
    for system in systems:
        for ex in targets:
            jobs.append((system, ex))

    records = await gather_capped(
        [generate_one(llm, ex, retriever, system) for system, ex in jobs],
        desc="generate",
    )
    _ = by_id
    return list(records)


async def score_all(
    llm: LLM,
    examples: Sequence[Example],
    records: Sequence[GenerationRecord],
    *,
    judge_model: str | None = None,
) -> list[ResponseScore]:
    by_id = {e.id: e for e in examples}

    async def one(rec: GenerationRecord) -> ResponseScore:
        ex = by_id[rec.example_id]
        if rec.error or not rec.reply.body.strip():
            return build_score(ex, rec, AuditResult(), _empty_verdict(), )
        try:
            a, v = await asyncio.gather(
                run_audit(llm, ex, rec.reply, model=judge_model),
                run_judge(llm, ex, rec.reply, model=judge_model),
            )
        except Exception as exc:  # noqa: BLE001
            rec = rec.model_copy(update={"error": "eval failed: " + str(exc)[:200]})
            return build_score(ex, rec, AuditResult(), _empty_verdict())
        return build_score(ex, rec, a, v)

    return list(await gather_capped([one(r) for r in records], desc="evaluate"))


def _empty_verdict() -> JudgeVerdict:
    from ..schemas import DimensionScore

    z = DimensionScore(score=0.0, reason="not evaluated")
    return JudgeVerdict(tone_fit=z, clarity=z, policy_safety=z, holistic=z,
                        would_send="do_not_send", biggest_problem="evaluation failed")


def save(run_dir: Path, name: str, rows: Sequence[object]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / (name + ".jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.model_dump() if hasattr(r, "model_dump") else r,
                                ensure_ascii=False) + "\n")
    return path


def load_scores(run_dir: Path) -> list[ResponseScore]:
    path = run_dir / "scores.jsonl"
    return [ResponseScore.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_generations(run_dir: Path) -> list[GenerationRecord]:
    path = run_dir / "generations.jsonl"
    return [GenerationRecord.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


DEFAULT_RUN = RUNS_DIR / "latest"
