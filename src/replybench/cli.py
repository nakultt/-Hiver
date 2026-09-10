"""replybench command line."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

from rich.console import Console
from rich.table import Table

from .config import DATA_DIR, RUNS_DIR, SETTINGS
from .llm import LLM
from .dataset.build import build_dataset, jaccard, load_dataset, _shingles
from .evaluate import aggregate, runner
from .evaluate.audit import audit as run_audit
from .evaluate.judge import judge as run_judge
from .evaluate.score import build_score
from .generate.generator import SYSTEMS, SYSTEM_DESCRIPTIONS, generate_one
from .generate.retrieve import Retriever
from .schemas import Email, Example, Reply
from .taxonomy import DIMENSIONS
from .validate import perturb

# legacy_windows=False stops rich falling back to a cp1252 writer, which
# cannot encode non-ASCII and crashes the command outright.
con = Console(legacy_windows=False)


def _warn_mock() -> None:
    if not SETTINGS.live:
        con.print("[bold yellow]LLM_BACKEND=mock[/] — plumbing only. "
                  "Every number below is filler. Set LLM_API_KEY for real results.")


# ---------------------------------------------------------------- dataset
def cmd_build_dataset(args: argparse.Namespace) -> None:
    _warn_mock()
    meta = asyncio.run(build_dataset(n_train=args.train, n_test=args.test, seed=args.seed))
    con.print(json.dumps(meta["counts"], indent=2))
    con.print("dropped as near-duplicates: " + str(len(meta["dropped_near_duplicates"])))
    con.print("[green]wrote[/] " + str(DATA_DIR / "dataset.jsonl"))


def cmd_dataset_report(args: argparse.Namespace) -> None:
    ex = load_dataset()
    t = Table(title="Dataset composition", show_lines=False)
    for c in ("split", "n", "median words (email)", "median words (reply)", "distinct intents"):
        t.add_column(c)
    for split in ("train", "test", "hard"):
        rows = [e for e in ex if e.split == split]
        if not rows:
            continue
        t.add_row(split, str(len(rows)),
                  str(int(median([len(r.incoming.body.split()) for r in rows]))),
                  str(int(median([len(r.reply.body.split()) for r in rows]))),
                  str(len({r.intent for r in rows})))
    con.print(t)

    # Balance across the sampling axes -- the claim the grid is supposed to support.
    for axis, keyfn in (("intent", lambda e: e.intent),
                        ("difficulty", lambda e: e.difficulty)):
        c = Counter(keyfn(e) for e in ex)
        con.print("[bold]" + axis + "[/]: " + ", ".join(k + "=" + str(v) for k, v in c.most_common()))

    # Leakage: worst test<->train similarity actually present in the shipped file.
    train_sh = [_shingles(e.incoming.body) for e in ex if e.split == "train"]
    worst = 0.0
    for e in ex:
        if e.split != "test":
            continue
        for s in train_sh:
            worst = max(worst, jaccard(_shingles(e.incoming.body), s))
    con.print("[bold]max test-train 5-gram Jaccard[/]: " + format(worst, ".3f")
              + "  (rows above 0.40 are dropped at build time)")

    # Lexical diversity of the replies -- catches a corpus written in one voice.
    toks = [w.lower() for e in ex for w in e.reply.body.split()]
    con.print("[bold]reply distinct-1[/]: " + format(len(set(toks)) / max(1, len(toks)), ".3f"))

    # Did the reply actually do what the scenario intended? Label-vs-intent gap.
    gaps = []
    for e in ex:
        want = set(e.provenance.get("intended_actions") or [])
        if want:
            got = set(e.reply.actions)
            gaps.append(len(want & got) / len(want))
    if gaps:
        con.print("[bold]intended-action recall in gold replies[/]: " + format(mean(gaps), ".3f")
                  + "  (1.0 would mean the writer always did exactly what the grid asked;"
                  " lower is expected and healthy)")

    hard = [e for e in ex if e.split == "hard"]
    con.print("[bold]hand-authored adversarial cases[/]: " + str(len(hard))
              + " — traps: " + ", ".join(sorted({h.trap.split("_")[0] for h in hard})))


# ---------------------------------------------------------------- suggest
def cmd_suggest(args: argparse.Namespace) -> None:
    _warn_mock()
    examples = load_dataset()
    body = Path(args.file).read_text(encoding="utf-8") if args.file else args.body
    if not body:
        con.print("[red]give --body or --file[/]"); sys.exit(2)

    ex = Example(
        id="adhoc", split="test", intent="unknown", difficulty="moderate",
        incoming=Email(subject=args.subject, body=body, sender_name=args.sender,
                       sender_email="customer@example.com", account_plan=args.plan),
        reply=Reply(body=""),
    )

    async def go() -> None:
        async with LLM() as llm:
            r = Retriever(runner.corpus(examples))
            gen = await generate_one(llm, ex, r, args.system)
            con.rule("SUGGESTED REPLY  (" + args.system + ")")
            con.print(gen.reply.body)
            con.rule("METADATA")
            con.print({"intent": gen.reply.intent, "actions": gen.reply.actions,
                       "cited_facts": gen.reply.cited_facts,
                       "confidence": gen.reply.confidence,
                       "needs_human_review": gen.reply.needs_human_review,
                       "open_questions": gen.reply.open_questions,
                       "retrieved_exemplars": gen.retrieved_ids})
            if args.explain:
                con.rule("ACCURACY (no reference reply, so action_match is not meaningful)")
                a = await run_audit(llm, ex, gen.reply)
                v = await run_judge(llm, ex, gen.reply)
                s = build_score(ex, gen, a, v)
                _print_score(s)

    asyncio.run(go())


def _print_score(s) -> None:
    t = Table(show_lines=False)
    t.add_column("dimension"); t.add_column("score", justify="right"); t.add_column("why")
    for k, d in s.dimensions.items():
        t.add_row(k, format(d.score, ".2f"), d.reason[:80])
    con.print(t)
    con.print("composite [bold]" + format(s.composite, ".3f") + "[/]  readiness [bold]"
              + s.readiness + "[/]")
    if s.hard_failures:
        con.print("[red]hard failures:[/] " + ", ".join(s.hard_failures))
    if s.warnings:
        con.print("[yellow]warnings:[/] " + ", ".join(s.warnings))
    for c in s.claims:
        if c.verdict in ("contradicted", "unsupported"):
            con.print("  [red]" + c.verdict + "[/]: " + c.claim[:90]
                      + ("  (" + c.note[:60] + ")" if c.note else ""))
    for a in s.asks:
        if a.status in ("ignored", "partially_answered"):
            con.print("  [yellow]" + a.status + "[/]: " + a.ask[:90])


# ---------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> None:
    _warn_mock()
    examples = load_dataset()
    systems = args.systems.split(",") if args.systems else list(SYSTEMS)
    out = RUNS_DIR / args.name

    async def go() -> None:
        async with LLM() as llm:
            gens = await runner.generate_all(llm, examples, systems, limit=args.limit)
            runner.save(out, "generations", gens)
            scores = await runner.score_all(llm, examples, gens)
            runner.save(out, "scores", scores)
            (out / "usage.json").write_text(
                json.dumps({"usage": llm.usage.as_dict(), "live_requests": llm.spent,
                            "models": {"gen": llm.s.gen_model, "judge": llm.s.judge_model}},
                           indent=2), encoding="utf-8")
        rep = aggregate.full_report(scores, examples)
        (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        con.print("[green]wrote[/] " + str(out))

    asyncio.run(go())
    cmd_report(argparse.Namespace(name=args.name))


# ---------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> None:
    out = RUNS_DIR / args.name
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    examples = load_dataset()
    scores = runner.load_scores(out)

    t = Table(title="Overall system score  (composite, 95% bootstrap CI)")
    t.add_column("system"); t.add_column("n", justify="right")
    t.add_column("composite", justify="right"); t.add_column("CI95")
    for d in DIMENSIONS:
        t.add_column(d[:9], justify="right")
    t.add_column("sendable", justify="right"); t.add_column("ROUGE-L", justify="right")
    order = sorted(rep["systems"], key=lambda s: -rep["systems"][s]["composite"])
    for s in order:
        r = rep["systems"][s]
        row = [s, str(r["n"]), format(r["composite"], ".3f"),
               "[" + format(r["composite_ci95"][0], ".2f") + ", "
               + format(r["composite_ci95"][1], ".2f") + "]"]
        row += [format(r["dimensions"].get(d, 0.0), ".2f") for d in DIMENSIONS]
        row += [format(r["sendable_rate"], ".2f"), format(r["rouge_l"], ".3f")]
        t.add_row(*row)
    con.print(t)

    con.print("\n[bold]System definitions[/]")
    for s in order:
        con.print("  " + s + ": " + SYSTEM_DESCRIPTIONS.get(s, ""))

    md = rep["metric_diagnostics"]
    con.print("\n[bold]Metric diagnostics[/]")
    con.print("  Spearman ROUGE-L vs composite (all systems): "
              + format(md["spearman_rouge_vs_composite_all"], ".3f"))
    con.print("  Spearman ROUGE-L vs composite (main only):   "
              + format(md["spearman_rouge_vs_composite_main"], ".3f"))
    con.print("  Spearman length-ratio vs composite (verbosity-bias probe): "
              + format(md["spearman_length_vs_composite"], ".3f"))

    con.print("\n[bold]Model-written gold vs human-written gold[/] "
              "(test minus hard; large positive = synthetic references flatter the system)")
    for s in order:
        gap = rep["systems"][s].get("model_gold_minus_human_gold")
        if gap is not None:
            con.print("  " + s.ljust(20) + format(gap, "+.3f"))

    main_fail = rep["systems"].get("main", {}).get("failures", {})
    if main_fail:
        con.print("\n[bold]main: what actually broke[/]")
        for k in ("hard_failures", "warnings", "most_missed_actions",
                  "most_over_claimed_actions", "traps_fallen_for"):
            if main_fail.get(k):
                con.print("  " + k + ": " + ", ".join(a + " x" + str(b) for a, b in main_fail[k][:5]))
        con.print("  ignored-ask rate: " + format(main_fail["ignored_ask_rate"], ".3f"))

    if args_show_worst := True:
        worst = sorted([s for s in scores if s.system == "main"], key=lambda s: s.composite)[:3]
        con.print("\n[bold]main: three worst responses[/]")
        by_id = {e.id: e for e in examples}
        for s in worst:
            ex = by_id.get(s.example_id)
            con.print("  " + s.example_id + "  composite=" + format(s.composite, ".2f")
                      + "  " + s.readiness
                      + ("  [dim]trap=" + ex.trap + "[/]" if ex and ex.trap else ""))
            if s.judge_note:
                con.print("     judge: " + s.judge_note[:110])


# ---------------------------------------------------------- validate-metric
def cmd_validate(args: argparse.Namespace) -> None:
    _warn_mock()
    examples = load_dataset()
    pool = [e for e in examples if e.split == "hard"] + [e for e in examples if e.split == "test"]
    out = RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)

    async def go() -> dict:
        async with LLM() as llm:
            cases = await perturb.build_cases(llm, pool, n=args.n)
            con.print("built " + str(len(cases)) + " perturbation cases from "
                      + str(args.n) + " gold replies")
            return await perturb.run_suite(llm, pool, cases)

    res = asyncio.run(go())
    (out / "metric_validation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    t = Table(title="Metric unit tests — inject one known defect, check the right dimension moves")
    t.add_column("defect"); t.add_column("n", justify="right"); t.add_column("targeted")
    t.add_column("d targeted", justify="right"); t.add_column("d composite", justify="right")
    t.add_column("d ROUGE-L", justify="right"); t.add_column("detected"); t.add_column("collateral")
    for kind, r in res.items():
        if kind.startswith("_"):
            continue
        tgt = r["targeted_dimensions"]
        dt = (format(mean([r["mean_delta_by_dimension"].get(d, 0.0) for d in tgt]), "+.3f")
              if tgt else "—")
        t.add_row(kind, str(r["n"]), ",".join(tgt) or "(none — control)", dt,
                  format(r["mean_delta_composite"], "+.3f"),
                  format(r["mean_delta_rouge_l"], "+.3f"),
                  {True: "yes", False: "NO", None: "n/a"}[r["detected"]],
                  ",".join(r["collateral_damage"]) or "—")
    con.print(t)
    s = res["_summary"]
    con.print("\n[bold]" + s["sensitivity"] + "[/]")
    con.print("clean detections (right dimension, no collateral): " + ", ".join(s["clean_detections"]))
    con.print("\n[bold]The paraphrase control[/] — same meaning, different words:")
    con.print("  composite moves " + str(s["paraphrase_composite_delta"])
              + "   ROUGE-L moves " + str(s["paraphrase_rouge_delta"]))
    con.print("  [dim]" + s["headline"] + "[/]")


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(prog="replybench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="generate the corpus")
    b.add_argument("--train", type=int, default=60)
    b.add_argument("--test", type=int, default=16)
    b.add_argument("--seed", type=int, default=20250911)
    b.set_defaults(func=cmd_build_dataset)

    d = sub.add_parser("dataset-report", help="composition, balance, leakage, diversity")
    d.set_defaults(func=cmd_dataset_report)

    s = sub.add_parser("suggest", help="suggest a reply to one email")
    s.add_argument("--body"); s.add_argument("--file")
    s.add_argument("--subject", default="(no subject)")
    s.add_argument("--sender", default="A Customer")
    s.add_argument("--plan", default="Growth")
    s.add_argument("--system", default="main", choices=list(SYSTEMS))
    s.add_argument("--explain", action="store_true", help="also score it")
    s.set_defaults(func=cmd_suggest)

    r = sub.add_parser("run", help="generate + score every system on the eval set")
    r.add_argument("--name", default="latest")
    r.add_argument("--systems", default="")
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="print a finished run")
    rep.add_argument("--name", default="latest")
    rep.set_defaults(func=cmd_report)

    v = sub.add_parser("validate-metric", help="prove the metric measures what it claims")
    v.add_argument("--name", default="latest")
    v.add_argument("--n", type=int, default=5)
    v.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
