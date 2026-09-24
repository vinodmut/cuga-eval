#!/usr/bin/env python3
"""Generate a self-contained, redacted HTML report for a native Hermes run.

The evaluation JSON contains simulated AppWorld passwords and bearer tokens in
tool traces.  This generator deliberately emits only action-level trace data,
redacts credential-shaped arguments, and never embeds raw tool results.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

HERMES_REVISION = "d6c9fb8ee8f54bc88897685ba422a60af2ecfab2"
APPWORLD_REVISION = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
CUGA_EVAL_BASE_REVISION = "2bc829fb248cdef9e1cbbdb47d5a9cdd8da94052"
MODEL = "openai/aws/claude-opus-5"
DEVELOPMENT_COST = 16.37523892
SENSITIVE_KEYS = re.compile(
    r"(^|_)(access_?token|refresh_?token|api_?key|authorization|password|passwd|secret)($|_)",
    re.IGNORECASE,
)
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
API_KEY = re.compile(r"\b(?:sk|sess)-[A-Za-z0-9_-]{12,}\b")

FAILURE_ANALYSIS = {
    "245cb43_1": (
        "Order API errors followed by the turn cap",
        "Hermes identified the only dish rack that fit, isolated it in the cart, and tried "
        "three payment-card combinations. AppWorld's place_order endpoint returned internal "
        "errors while leaving three empty orders behind. Hermes stopped at the 25-turn limit "
        "without a valid order or completion call.",
    ),
    "4d12842_1": (
        "Turn cap during a long deletion pass",
        "Hermes found 22 archived threads before the month boundary and reported deleting 16. "
        "The one-call-per-thread sequence exhausted the configured turn budget before the set "
        "was complete.",
    ),
    "ba46d91_1": (
        "Finished normally with the wrong rounded answer",
        "Hermes submitted 336. The official answer assertion failed while the no-model-change "
        "assertion passed, so this is a reasoning or date-boundary error rather than a runtime "
        "or state-mutation failure.",
    ),
    "5238afc_1": (
        "Turn cap while isolating cart products",
        "Hermes identified four weightlifting benches and began moving unrelated cart items to "
        "the wish list. It reached the turn limit before placing the order, restoring parked "
        "items, or calling complete_task.",
    ),
    "0d22252_1": (
        "Order API errors followed by the turn cap",
        "Hermes found the two wrench sets and attempted checkout three times. Each place_order "
        "call returned an internal error but created a broken order with no line items. The cart "
        "was left modified and the task was not completed.",
    ),
    "277d81d_1": (
        "One thread short when the turn cap was reached",
        "Hermes enumerated 16 distinct unread threads before the cutoff and reported marking 15 "
        "as read. The final thread and completion call remained when the 25-turn budget ended.",
    ),
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def redact(value: Any, key: str = "") -> Any:
    if SENSITIVE_KEYS.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, str):
        return API_KEY.sub("[REDACTED_API_KEY]", JWT.sub("[REDACTED_TOKEN]", value))
    return value


def compact_json(value: Any) -> str:
    return json.dumps(redact(value), ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


def fmt_int(value: Any) -> str:
    return f"{int(value or 0):,}"


def fmt_duration(seconds: Any) -> str:
    total = max(0, round(float(seconds or 0)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def artifact_path(value: str) -> str:
    marker = "/benchmarks/appworld/"
    if marker in value:
        return "benchmarks/appworld/" + value.split(marker, 1)[1]
    return value


def trace_lines(result: dict[str, Any]) -> str:
    """Render safe action-level evidence without raw MCP response bodies."""
    lines: list[str] = []
    sequence = 0
    for wrapper in result.get("tool_calls") or []:
        wrapper_name = str(wrapper.get("name") or "unknown")
        arguments = wrapper.get("arguments") or {}
        raw_result = str(wrapper.get("result") or "")
        result_note = ""
        if "Internal Server Error" in raw_result:
            result_note = "  -> result contained Internal Server Error"
        elif raw_result:
            result_note = f"  -> result retained locally ({len(raw_result):,} characters; omitted)"

        if wrapper_name == "tool_call" and isinstance(arguments, dict):
            nested = arguments.get("calls") or []
            for call in nested:
                if not isinstance(call, dict):
                    continue
                sequence += 1
                name = str(call.get("name") or "unknown")
                safe_args = compact_json(call.get("arguments") or {})
                lines.append(f"{sequence:02d}. {name} {safe_args}{result_note}")
        else:
            sequence += 1
            lines.append(f"{sequence:02d}. {wrapper_name} {compact_json(arguments)}{result_note}")
    return "\n".join(lines) or "No exported tool calls."


def check_list(items: list[dict[str, Any]], empty_text: str) -> str:
    if not items:
        return f'<p class="empty">{esc(empty_text)}</p>'
    return "<ol>" + "".join(f"<li>{esc(item.get('requirement', ''))}</li>" for item in items) + "</ol>"


def stat(label: str, value: str, note: str) -> str:
    return (
        f'<div class="stat"><span>{esc(label)}</span><strong>{esc(value)}</strong>'
        f"<small>{esc(note)}</small></div>"
    )


def result_rows(results: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for index, result in enumerate(results, 1):
        evaluation = result.get("appworld_evaluation") or {}
        passed = bool(result.get("success"))
        status = "pass" if passed else "fail"
        label = "PASS" if passed else "FAIL"
        task = str(result.get("task_name") or "")
        instruction = str(result.get("intent") or "")
        search = f"{task} {instruction} {label} {result.get('response') or ''}".lower()
        percent = float(evaluation.get("pass_percentage") or 0)
        rows.append(
            f"""
        <tr class="result-row" data-status="{status}" data-search="{esc(search)}">
          <td><span class="row-num">{index:02d}</span></td>
          <td><a href="#task-{esc(task)}" class="task-link">{esc(task)}</a></td>
          <td class="instruction-cell">{esc(instruction)}</td>
          <td><span class="badge {status}">{label}</span></td>
          <td><strong>{int(evaluation.get('pass_count') or 0)}/{int(evaluation.get('total_count') or 0)}</strong><div class="mini-bar"><i style="width:{percent:.1f}%"></i></div></td>
          <td>{esc(fmt_duration(result.get('full_execution_time')))}</td>
          <td class="num">{fmt_int(result.get('total_llm_calls'))}</td>
          <td class="num">{fmt_int(result.get('input_tokens'))}</td>
          <td class="num">{fmt_int(result.get('output_tokens'))}</td>
          <td><a href="#task-{esc(task)}" class="details-link">Inspect</a></td>
        </tr>"""
        )
    return "".join(rows)


def task_cards(results: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for index, result in enumerate(results, 1):
        evaluation = result.get("appworld_evaluation") or {}
        passed = bool(result.get("success"))
        status = "pass" if passed else "fail"
        label = "PASS" if passed else "FAIL"
        task = str(result.get("task_name") or "")
        passes = evaluation.get("passes") or []
        failures = evaluation.get("failures") or []
        analysis = "<!-- No failure analysis: every verifier assertion passed. -->"
        if not passed:
            title, body = FAILURE_ANALYSIS.get(
                task,
                ("Verifier failure", "The task did not satisfy every official AppWorld assertion."),
            )
            analysis = (
                '<div class="failure-analysis"><div><span class="eyebrow">Observed failure mode</span>'
                f"<strong>{esc(title)}</strong></div><p>{esc(body)}</p></div>"
            )
        artifacts = result.get("artifacts") or {}
        artifact_items = "".join(
            f"<li><b>{esc(name)}</b><code>{esc(artifact_path(str(path)))}</code></li>"
            for name, path in artifacts.items()
        )
        runtime_status = result.get("error") or "Clean exit"
        cards.append(
            f"""
      <details class="task-card" id="task-{esc(task)}" data-status="{status}" data-search="{esc((task + ' ' + str(result.get('intent') or '') + ' ' + label).lower())}">
        <summary>
          <span class="row-num">{index:02d}</span>
          <span class="task-summary-copy"><strong>{esc(task)}</strong><span>{esc(result.get('intent'))}</span></span>
          <span class="summary-score">{int(evaluation.get('pass_count') or 0)}/{int(evaluation.get('total_count') or 0)} checks</span>
          <span class="badge {status}">{label}</span><span class="chevron">⌄</span>
        </summary>
        <div class="task-body">
          {analysis}
          <div class="metric-strip">
            <div><span>Verifier</span><strong>{float(evaluation.get('pass_percentage') or 0):.1f}%</strong></div>
            <div><span>Session steps</span><strong>{fmt_int(result.get('steps'))}</strong></div>
            <div><span>LLM calls</span><strong>{fmt_int(result.get('total_llm_calls'))}</strong></div>
            <div><span>Runtime</span><strong>{esc(fmt_duration(result.get('full_execution_time')))}</strong></div>
            <div><span>Total tokens</span><strong>{fmt_int(result.get('total_tokens'))}</strong></div>
            <div><span>Estimated cost</span><strong>${float(result.get('total_cost') or 0):.4f}</strong></div>
          </div>
          <div class="two-col">
            <section><h4>Instruction</h4><blockquote>{esc(result.get('intent'))}</blockquote></section>
            <section><h4>Run identity</h4><dl class="compact-dl">
              <div><dt>Difficulty</dt><dd>{esc(result.get('difficulty'))}</dd></div>
              <div><dt>Harness thread</dt><dd><code>{esc(result.get('thread_id'))}</code></dd></div>
              <div><dt>Hermes session</dt><dd><code>{esc(result.get('hermes_session_id'))}</code></dd></div>
              <div><dt>Runtime status</dt><dd>{esc(runtime_status)}</dd></div>
            </dl></section>
          </div>
          <h4>Hermes final output</h4>
          <pre class="agent-output">{esc(redact(result.get('response') or ''))}</pre>
          <div class="usage-grid">
            <span>Input <b>{fmt_int(result.get('input_tokens'))}</b></span>
            <span>Output <b>{fmt_int(result.get('output_tokens'))}</b></span>
            <span>Cache reads <b>{fmt_int(result.get('cache_read_tokens'))}</b></span>
            <span>Cache writes <b>{fmt_int(result.get('cache_write_tokens'))}</b></span>
            <span>Reasoning <b>{fmt_int(result.get('reasoning_tokens'))}</b></span>
            <span>Cost basis <b>{esc(result.get('cost_status'))}</b></span>
          </div>
          <div class="checks-grid">
            <section class="checks passed-checks"><h4>Passed assertions <span>{len(passes)}</span></h4>{check_list(passes, 'None')}</section>
            <section class="checks failed-checks"><h4>Failed assertions <span>{len(failures)}</span></h4>{check_list(failures, 'None — every verifier assertion passed.')}</section>
          </div>
          <details class="trace"><summary>Redacted native Hermes action trace ({len(result.get('tool_calls') or [])} top-level calls)</summary><pre>{esc(trace_lines(result))}</pre></details>
          <details class="prompt-shape"><summary>Prompt and isolation evidence</summary>
            <div class="prompt-note"><p>The per-task prompt combined the shared AppWorld policy, a native-completion extension, generated user context, and the instruction above. Private user context is intentionally omitted.</p>
            <pre>toolsets: [appworld]\nplatform_toolsets.cli: [appworld]\nagent.max_turns: 25\nmemory.memory_enabled: false\nmemory.user_profile_enabled: false\ncheckpoints.enabled: false\nmcp_servers.appworld.url: http://127.0.0.1:10000/mcp/</pre></div>
          </details>
          <h4 class="artifact-heading">Retained local evidence <span class="subtle">gitignored; may contain simulated credentials</span></h4>
          <ul class="artifacts">{artifact_items}</ul>
        </div>
      </details>"""
        )
    return "".join(cards)


def failure_cards(results: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for result in results:
        if result.get("success"):
            continue
        evaluation = result.get("appworld_evaluation") or {}
        task = str(result.get("task_name") or "")
        title, body = FAILURE_ANALYSIS.get(task, ("Verifier failure", "See task evidence."))
        runtime = "25-turn boundary" if result.get("error") else "clean runtime exit"
        cards.append(
            f"""
        <a class="failure-card" href="#task-{esc(task)}">
          <div class="failure-head"><code>{esc(task)}</code><span class="badge fail">FAIL</span></div>
          <strong>{esc(title)}</strong><p>{esc(body)}</p>
          <div class="failure-meta"><span>{int(evaluation.get('pass_count') or 0)}/{int(evaluation.get('total_count') or 0)} checks</span><span>{esc(runtime)}</span></div>
        </a>"""
        )
    return "".join(cards)


def signed_number(value: float, *, suffix: str = "", digits: int = 2) -> str:
    if value == 0:
        return f"0{suffix}"
    return f"{value:+.{digits}f}{suffix}"


def harbor_comparison(
    results: list[dict[str, Any]], harbor: dict[str, Any] | None, harbor_source: str | None
) -> str:
    if not harbor:
        return ""

    harbor_results = harbor["results"]
    harbor_usage = harbor["usage"]
    harbor_trials = {str(trial["task_id"]).replace("-", "_"): trial for trial in harbor.get("trials") or []}
    cuga_cost = sum(float(item.get("total_cost") or 0) for item in results)
    cuga_agent_seconds = sum(float(item.get("full_execution_time") or 0) for item in results)
    harbor_agent_seconds = sum(float(item.get("agent_execution_sec") or 0) for item in harbor_trials.values())
    cuga_cache_reads = sum(int(item.get("cache_read_tokens") or 0) for item in results)
    cuga_cache_writes = sum(int(item.get("cache_write_tokens") or 0) for item in results)
    cuga_calls = sum(int(item.get("total_llm_calls") or 0) for item in results)
    cuga_input = sum(int(item.get("input_tokens") or 0) for item in results)
    cuga_output = sum(int(item.get("output_tokens") or 0) for item in results)
    harbor_cost = float(harbor["cost_estimate_usd"]["with_cache_read_at_0.10x_and_write_at_1.25x"])

    def percent_delta(current: float, baseline: float) -> str:
        return signed_number((current / baseline - 1) * 100, suffix="%") if baseline else "n/a"

    metric_rows = [
        (
            "Binary task success",
            f"{sum(bool(item.get('success')) for item in results)}/24 ({sum(bool(item.get('success')) for item in results) / 24 * 100:.2f}%)",
            f"{harbor_results['task_successes']}/24 ({float(harbor_results['tgc']) * 100:.2f}%)",
            f"+1 task ({signed_number((sum(bool(item.get('success')) for item in results) / 24 - float(harbor_results['tgc'])) * 100, suffix=' pp')})",
            "good",
        ),
        (
            "Mean task test pass rate",
            f"{sum(float(item.get('match_rate') or 0) for item in results) / 24 * 100:.2f}%",
            f"{float(harbor_results['average_test_pass_rate']) * 100:.2f}%",
            signed_number(
                (
                    sum(float(item.get("match_rate") or 0) for item in results) / 24
                    - float(harbor_results["average_test_pass_rate"])
                )
                * 100,
                suffix=" pp",
            ),
            "good",
        ),
        (
            "Non-zero agent exits",
            str(sum(bool(item.get("error")) for item in results)),
            str(harbor_results["nonzero_agent_exits"]),
            "0",
            "neutral",
        ),
        (
            "LLM calls",
            fmt_int(cuga_calls),
            fmt_int(harbor_usage["api_calls"]),
            signed_number(cuga_calls - int(harbor_usage["api_calls"]), digits=0),
            "neutral",
        ),
        (
            "Input tokens",
            fmt_int(cuga_input),
            fmt_int(harbor_usage["input_tokens"]),
            percent_delta(cuga_input, float(harbor_usage["input_tokens"])),
            "neutral",
        ),
        (
            "Output tokens",
            fmt_int(cuga_output),
            fmt_int(harbor_usage["output_tokens"]),
            percent_delta(cuga_output, float(harbor_usage["output_tokens"])),
            "neutral",
        ),
        (
            "Cache-read tokens",
            fmt_int(cuga_cache_reads),
            fmt_int(harbor_usage["cache_read_tokens"]),
            percent_delta(cuga_cache_reads, float(harbor_usage["cache_read_tokens"])),
            "neutral",
        ),
        (
            "Cache-write tokens",
            fmt_int(cuga_cache_writes),
            fmt_int(harbor_usage["cache_write_tokens"]),
            percent_delta(cuga_cache_writes, float(harbor_usage["cache_write_tokens"])),
            "neutral",
        ),
        (
            "Summed agent execution",
            fmt_duration(cuga_agent_seconds),
            fmt_duration(harbor_agent_seconds),
            percent_delta(cuga_agent_seconds, harbor_agent_seconds),
            "neutral",
        ),
        (
            "Estimated cost",
            f"${cuga_cost:.4f}",
            f"${harbor_cost:.4f}",
            f"{'-' if cuga_cost < harbor_cost else '+'}${abs(cuga_cost - harbor_cost):.4f} "
            f"({percent_delta(cuga_cost, harbor_cost)})",
            "neutral",
        ),
    ]
    metrics_html = "".join(
        f'<tr><td><strong>{esc(label)}</strong></td><td>{esc(cuga)}</td><td>{esc(old)}</td>'
        f'<td><span class="delta {kind}">{esc(delta)}</span></td></tr>'
        for label, cuga, old, delta, kind in metric_rows
    )

    task_rows: list[str] = []
    improvements = 0
    regressions = 0
    agreements = 0
    for result in results:
        task = str(result.get("task_name") or "")
        trial = harbor_trials[task]
        cuga_pass = bool(result.get("success"))
        harbor_pass = float(trial.get("reward") or 0) == 1.0
        if cuga_pass == harbor_pass:
            outcome = "Same"
            kind = "neutral"
            agreements += 1
        elif cuga_pass:
            outcome = "CUGA +1"
            kind = "good"
            improvements += 1
        else:
            outcome = "Harbor +1"
            kind = "bad"
            regressions += 1
        cuga_eval = result.get("appworld_evaluation") or {}
        cuga_score = float(cuga_eval.get("pass_percentage") or 0)
        harbor_score = float(trial.get("test_pass_rate") or 0) * 100
        score_delta = cuga_score - harbor_score
        task_rows.append(
            f"""
          <tr>
            <td><a href="#task-{esc(task)}" class="task-link">{esc(task)}</a></td>
            <td class="instruction-cell">{esc(result.get('intent'))}</td>
            <td><span class="badge {'pass' if cuga_pass else 'fail'}">{'PASS' if cuga_pass else 'FAIL'}</span><br><small>{cuga_score:.1f}% tests</small></td>
            <td><span class="badge {'pass' if harbor_pass else 'fail'}">{'PASS' if harbor_pass else 'FAIL'}</span><br><small>{harbor_score:.1f}% tests</small></td>
            <td><span class="delta {kind}">{outcome}</span><br><small>{signed_number(score_delta, suffix=' pp')}</small></td>
          </tr>"""
        )

    source = harbor_source or "Harbor run-summary.json"
    return f"""
    <section class="section" id="comparison">
      <div class="section-head"><h2>Comparison with Harbor</h2><p>Both runs use the same 24 task IDs, model identifier, AppWorld revision, and 25-turn limit. CUGA Eval passed one additional task, but Harbor did not record its exact Hermes revision.</p></div>
      <div class="comparison-lead">
        <div><span>Binary result</span><strong>+1 task</strong><small>18/24 here versus 17/24 in Harbor</small></div>
        <div><span>Outcome agreement</span><strong>{agreements}/24</strong><small>{agreements / 24 * 100:.1f}% of tasks had the same binary result</small></div>
        <div><span>Changed outcomes</span><strong>{improvements} ↑ · {regressions} ↓</strong><small><code>9bf2c8a_1</code> was the only change</small></div>
        <div><span>Estimated cost</span><strong>{percent_delta(cuga_cost, harbor_cost)}</strong><small>${cuga_cost:.2f} here versus ${harbor_cost:.2f} in Harbor</small></div>
      </div>
      <div class="panel comparison-note"><p><strong>The actionable difference:</strong> <code>9bf2c8a_1</code> (move all food processors from the Amazon cart to the wish list) improved from 3/6 verifier checks and failure in Harbor to 6/6 and success here. The other 23 binary outcomes matched. Among shared failures, <code>277d81d_1</code> had lower partial credit here (4/6 versus 5/6); the remaining five failure scores were the same.</p></div>
      <h3>Aggregate comparison</h3>
      <div class="table-wrap"><table class="results-table comparison-table"><thead><tr><th>Metric</th><th>CUGA Eval</th><th>Harbor</th><th>CUGA − Harbor</th></tr></thead><tbody>{metrics_html}</tbody></table></div>
      <p class="comparison-caption">Harbor wall time was {fmt_duration(harbor.get('wall_time_sec'))} at concurrency 2. The CUGA Eval run window was 39m 21s with sequential execution; those wall-clock values include different orchestration overhead and should not be read as a direct speed benchmark. The summed native-agent execution comparison above is narrower, but still not controlled.</p>
      <h3>Task-by-task outcomes</h3>
      <div class="table-wrap"><table class="results-table comparison-table task-comparison"><thead><tr><th>Task</th><th>Instruction</th><th>CUGA Eval</th><th>Harbor</th><th>Difference</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table></div>
      <div class="callout"><strong>Do not read the +1 task as a causal harness win</strong><p>This is one attempt per task. Harbor did not record an exact Hermes revision, while CUGA Eval pinned Hermes <code>{HERMES_REVISION[:12]}</code>; prompts, isolation, scheduling, and completion plumbing also differ. The comparison demonstrates broadly similar behavior and identifies where outcomes diverged, not why.</p></div>
      <p class="source-note"><strong>Harbor evidence:</strong> <code>{esc(source)}</code>; job <code>{esc(harbor.get('job_id'))}</code>, run September 23, 2026 from 18:30:10 to 19:29:14 local time.</p>
    </section>"""


def generate(
    data: dict[str, Any],
    source_name: str,
    harbor: dict[str, Any] | None = None,
    harbor_source: str | None = None,
) -> str:
    metrics = data["metrics"]
    results = data["results"]
    passed = int(metrics["passed"])
    failed = int(metrics["failed"])
    total = int(metrics["total_tasks"])
    input_tokens = int(metrics["total_input_tokens"])
    output_tokens = int(metrics["total_output_tokens"])
    cache_reads = sum(int(item.get("cache_read_tokens") or 0) for item in results)
    cache_writes = sum(int(item.get("cache_write_tokens") or 0) for item in results)
    run_cost = sum(float(item.get("total_cost") or 0) for item in results)
    agent_time = sum(float(item.get("full_execution_time") or 0) for item in results)
    runtime_failures = sum(bool(item.get("error")) for item in results)
    checks_passed = sum(
        int((item.get("appworld_evaluation") or {}).get("pass_count") or 0) for item in results
    )
    checks_total = sum(
        int((item.get("appworld_evaluation") or {}).get("total_count") or 0) for item in results
    )
    pass_pct = float(metrics["pass_rate"]) * 100
    mean_pct = float(metrics["avg_match_rate"]) * 100
    rows = result_rows(results)
    cards = task_cards(results)
    failures_html = failure_cards(results)
    comparison_html = harbor_comparison(results, harbor, harbor_source)
    comparison_nav = '<a href="#comparison">Harbor comparison</a>' if harbor else ""

    css = r"""
:root{--ink:#18251f;--muted:#617069;--paper:#f6f3eb;--card:#fffdf7;--line:#d9ded7;--green:#176b4b;--green-soft:#e0f1e8;--red:#b63a2b;--red-soft:#f8e4df;--amber:#a86413;--amber-soft:#fff0d3;--navy:#173f4f;--teal:#2c7c75;--gold:#d4a24c;--shadow:0 14px 40px rgba(26,45,37,.08)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.34;background-image:radial-gradient(#b9beb7 .7px,transparent .7px);background-size:14px 14px;z-index:-1}a{color:var(--navy)}code,pre,.mono{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace}.wrap{width:min(1200px,calc(100% - 36px));margin:0 auto}.hero{position:relative;overflow:hidden;background:var(--navy);color:#fff;padding:72px 0 64px;border-bottom:7px solid var(--gold)}.hero:after{content:"";position:absolute;right:-140px;top:-260px;width:620px;height:620px;border:100px solid rgba(255,255,255,.055);border-radius:50%}.kicker{display:inline-flex;gap:9px;align-items:center;text-transform:uppercase;letter-spacing:.14em;font-weight:800;font-size:.75rem;color:#f0c875}.kicker:before{content:"";width:28px;height:2px;background:#f0c875}h1{font-family:Georgia,"Times New Roman",serif;font-weight:600;font-size:clamp(2.7rem,6vw,5.25rem);line-height:.96;letter-spacing:-.045em;margin:18px 0 22px;max-width:900px}.hero-copy{font-size:1.15rem;color:#dbe8e8;max-width:800px;margin:0}.hero-copy code{font-size:.9em}.hero-meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:30px}.hero-meta span{border:1px solid rgba(255,255,255,.24);padding:7px 11px;border-radius:999px;font-size:.83rem;background:rgba(255,255,255,.06)}nav{position:sticky;top:0;z-index:20;background:rgba(246,243,235,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}nav .wrap{display:flex;gap:24px;overflow:auto;padding:12px 0}nav a{white-space:nowrap;text-decoration:none;font-size:.8rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}nav a:hover{color:var(--ink)}main{padding:46px 0 80px}section[id]{scroll-margin-top:70px}.section{margin:0 0 64px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:24px}.section-head h2{font-family:Georgia,serif;font-size:2.25rem;letter-spacing:-.03em;margin:0}.section-head p{max-width:630px;color:var(--muted);margin:0}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.stat span{display:block;color:var(--muted);font-size:.76rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.stat strong{display:block;font-family:Georgia,serif;font-size:2rem;line-height:1.1;margin:8px 0 3px}.stat small{color:var(--muted)}.result-band{display:flex;height:26px;border-radius:999px;overflow:hidden;margin:22px 0 12px;background:#ddd}.result-band .ok{background:var(--green)}.result-band .bad{background:var(--red)}.legend{display:flex;gap:20px;color:var(--muted);font-size:.85rem}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:7px}.dot.ok{background:var(--green)}.dot.bad{background:var(--red)}.callout{border-left:5px solid var(--amber);background:var(--amber-soft);padding:18px 22px;border-radius:0 12px 12px 0;margin:22px 0}.callout strong{display:block;margin-bottom:4px}.callout p{margin:0;color:#684515}.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:var(--shadow)}.config-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0}.config-grid div{padding:14px 16px;border-bottom:1px solid var(--line)}.config-grid dt{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:800}.config-grid dd{margin:5px 0 0;font-weight:650;overflow-wrap:anywhere}.config-grid code{font-size:.78rem}.provenance{width:100%;border-collapse:collapse}.provenance th,.provenance td{padding:14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}.provenance th{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.provenance code{font-size:.78rem;overflow-wrap:anywhere}.provenance tr:last-child td{border-bottom:0}.architecture{display:grid;grid-template-columns:1fr auto 1.12fr auto 1.15fr auto 1.05fr;align-items:center;gap:12px;margin-top:20px}.node{padding:18px;border:1px solid var(--line);border-radius:13px;background:#fff}.node b{display:block;margin-bottom:4px}.node small{color:var(--muted)}.arrow{font-size:1.5rem;color:var(--teal)}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}.comparison article{border:1px solid var(--line);background:var(--card);padding:18px;border-radius:12px}.comparison h4{margin:0 0 7px}.comparison p{margin:0;color:var(--muted)}.comparison-lead{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}.comparison-lead div{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.comparison-lead span{display:block;color:var(--muted);font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.comparison-lead strong{display:block;font-family:Georgia,serif;font-size:1.8rem;margin:7px 0}.comparison-lead small{color:var(--muted)}.comparison-note{margin-bottom:22px}.comparison-note p,.comparison-caption,.source-note{margin:0;color:var(--muted)}.comparison-caption,.source-note{font-size:.84rem;margin-top:12px}.comparison-table{min-width:760px}.task-comparison{min-width:930px}.delta{font-weight:850}.delta.good{color:var(--green)}.delta.bad{color:var(--red)}.delta.neutral{color:var(--muted)}.cost-grid{display:grid;grid-template-columns:1.08fr 1fr;gap:18px}.cost-scenarios{display:grid;gap:10px}.cost-row{display:flex;justify-content:space-between;align-items:baseline;padding:14px;border:1px solid var(--line);border-radius:10px}.cost-row strong{font-family:Georgia,serif;font-size:1.55rem}.token-bars{display:grid;gap:13px}.token-bar .line{height:10px;background:#e6e6df;border-radius:8px;overflow:hidden}.token-bar i{display:block;height:100%;background:var(--teal)}.token-bar .labels{display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:5px}.token-bar .labels span{color:var(--muted)}.failure-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.failure-card{display:block;text-decoration:none;color:var(--ink);background:var(--card);border:1px solid #e3c1b9;border-radius:14px;padding:18px;transition:.18s}.failure-card:hover{transform:translateY(-2px);box-shadow:var(--shadow)}.failure-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}.failure-card>strong{display:block}.failure-card p{color:var(--muted);margin:8px 0}.failure-meta{display:flex;gap:10px;flex-wrap:wrap}.failure-meta span{background:#f1eee6;padding:4px 8px;border-radius:7px;font-size:.72rem}.filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}.filter-btn{border:1px solid var(--line);background:var(--card);padding:8px 12px;border-radius:999px;font-weight:750;color:var(--muted);cursor:pointer}.filter-btn.active{background:var(--navy);border-color:var(--navy);color:#fff}.search{margin-left:auto;min-width:280px;border:1px solid var(--line);border-radius:9px;padding:9px 12px;background:#fff;font:inherit}.table-wrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}.results-table{width:100%;border-collapse:collapse;min-width:1050px}.results-table th{position:sticky;top:0;background:#ebe8df;z-index:1;text-align:left;padding:11px 10px;font-size:.69rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.results-table td{padding:11px 10px;border-top:1px solid var(--line);vertical-align:top;font-size:.83rem}.results-table tr:hover td{background:#faf8f2}.instruction-cell{min-width:360px}.row-num{font-family:Georgia,serif;color:#9b8e77;font-weight:700}.task-link{font-family:"SFMono-Regular",monospace;font-weight:800;text-decoration:none}.details-link{font-weight:800}.num{text-align:right;font-variant-numeric:tabular-nums}.badge{display:inline-flex;align-items:center;justify-content:center;border-radius:999px;padding:4px 8px;font-size:.66rem;line-height:1;font-weight:900;letter-spacing:.08em}.badge.pass{color:var(--green);background:var(--green-soft)}.badge.fail{color:var(--red);background:var(--red-soft)}.mini-bar{width:65px;height:4px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:5px}.mini-bar i{display:block;height:100%;background:var(--teal)}.task-list{display:grid;gap:10px;margin-top:20px}.task-card{background:var(--card);border:1px solid var(--line);border-radius:13px;overflow:hidden;scroll-margin-top:75px}.task-card[open]{box-shadow:var(--shadow)}.task-card>summary{display:grid;grid-template-columns:32px minmax(0,1fr) auto auto 22px;align-items:center;gap:14px;padding:15px 18px;cursor:pointer;list-style:none}.task-card>summary::-webkit-details-marker{display:none}.task-summary-copy{min-width:0}.task-summary-copy strong{display:block;font-family:"SFMono-Regular",monospace;font-size:.83rem}.task-summary-copy span{display:block;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.84rem}.summary-score{font-size:.78rem;color:var(--muted)}.chevron{font-size:1.25rem;transition:.2s}.task-card[open] .chevron{transform:rotate(180deg)}.task-body{border-top:1px solid var(--line);padding:22px}.task-body h4{margin:0 0 10px;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em}.failure-analysis{display:grid;grid-template-columns:240px 1fr;gap:20px;background:var(--red-soft);border:1px solid #edc4ba;padding:16px;border-radius:11px;margin-bottom:18px}.failure-analysis .eyebrow{display:block;text-transform:uppercase;letter-spacing:.07em;font-size:.66rem;color:var(--red);font-weight:900}.failure-analysis p{margin:0}.metric-strip{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:20px}.metric-strip div{padding:11px;border-right:1px solid var(--line)}.metric-strip div:last-child{border:0}.metric-strip span{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em}.metric-strip strong{display:block;font-size:.9rem;margin-top:3px}.two-col,.checks-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}.two-col section,.checks{min-width:0}.two-col blockquote{margin:0;border-left:3px solid var(--teal);padding:12px 15px;background:#edf4f1;border-radius:0 8px 8px 0}.compact-dl{margin:0}.compact-dl div{display:flex;justify-content:space-between;gap:20px;padding:7px 0;border-bottom:1px solid var(--line)}.compact-dl dt{color:var(--muted)}.compact-dl dd{margin:0;text-align:right;overflow-wrap:anywhere}.usage-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0 20px}.usage-grid span{padding:9px 11px;background:#f0eee7;border-radius:8px;font-size:.78rem}.usage-grid b{float:right}.checks{border:1px solid var(--line);border-radius:10px;padding:15px}.checks h4 span{float:right;border-radius:999px;padding:2px 7px;background:#eee}.checks ol{padding-left:20px;margin:0}.checks li{padding:6px 0;font-size:.82rem}.failed-checks{border-color:#edc4ba}.empty{color:var(--muted);font-size:.82rem;margin:0}.trace summary,.prompt-shape summary{cursor:pointer;font-weight:750;color:var(--navy);padding:9px 0}.trace pre,.prompt-note pre,.agent-output,.codeblock{white-space:pre-wrap;overflow:auto;background:#132a31;color:#dfeae8;border-radius:10px;padding:15px;font-size:.74rem;line-height:1.55}.trace pre{max-height:420px}.agent-output{max-height:420px}.prompt-note p{color:var(--muted)}.artifact-heading{margin-top:18px!important}.artifacts{padding-left:20px;margin:0}.artifacts li{margin:5px 0}.artifacts b{display:inline-block;width:75px}.artifacts code{font-size:.75rem;overflow-wrap:anywhere}.subtle{color:var(--muted);font-size:.68rem;text-transform:none;letter-spacing:0;font-weight:500}.steps{counter-reset:step;display:grid;gap:14px}.step{position:relative;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 22px 22px 74px}.step:before{counter-increment:step;content:counter(step);position:absolute;left:20px;top:20px;width:35px;height:35px;border-radius:50%;display:grid;place-items:center;background:var(--navy);color:#fff;font-family:Georgia,serif;font-weight:800}.step h3{margin:0 0 7px}.step p{margin:5px 0;color:var(--muted)}.codeblock{color:#e8f2ef;margin:12px 0;max-height:520px}.limits{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.limit{padding:18px;background:var(--card);border:1px solid var(--line);border-radius:12px}.limit strong{display:block;margin-bottom:5px}.limit p{margin:0;color:var(--muted)}.foot{border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:.8rem}.hidden{display:none!important}
@media(max-width:1000px){.cards{grid-template-columns:repeat(3,1fr)}.config-grid{grid-template-columns:repeat(2,1fr)}.comparison-lead{grid-template-columns:repeat(2,1fr)}.architecture{grid-template-columns:1fr}.arrow{transform:rotate(90deg);text-align:center}.metric-strip{grid-template-columns:repeat(3,1fr)}.cost-grid{grid-template-columns:1fr}}
@media(max-width:680px){.wrap{width:min(100% - 22px,1200px)}.hero{padding:52px 0}.cards{grid-template-columns:repeat(2,1fr)}.section-head{display:block}.section-head p{margin-top:8px}.config-grid,.failure-grid,.comparison,.comparison-lead,.two-col,.checks-grid,.limits{grid-template-columns:1fr}.task-card>summary{grid-template-columns:25px 1fr auto 18px}.summary-score{display:none}.metric-strip,.usage-grid{grid-template-columns:repeat(2,1fr)}.failure-analysis{grid-template-columns:1fr}.search{width:100%;min-width:0;margin-left:0}}
@media print{nav,.filters{display:none}.hero{padding:30px 0}.task-card{break-inside:avoid}.task-card:not([open]) .task-body{display:block}.wrap{width:100%}body{background:#fff}}
"""

    script = r"""
const buttons=[...document.querySelectorAll('[data-filter]')];
const search=document.getElementById('task-search');
const rows=[...document.querySelectorAll('.result-row')];
const cards=[...document.querySelectorAll('.task-card')];
let active='all';
function refresh(){
  const query=search.value.trim().toLowerCase();let visible=0;
  rows.forEach((row,index)=>{const match=(active==='all'||row.dataset.status===active)&&row.dataset.search.includes(query);row.classList.toggle('hidden',!match);cards[index].classList.toggle('hidden',!match);if(match)visible+=1;});
  document.getElementById('visible-count').textContent=String(visible);
}
buttons.forEach(button=>button.addEventListener('click',()=>{active=button.dataset.filter;buttons.forEach(item=>item.classList.toggle('active',item===button));refresh();}));
search.addEventListener('input',refresh);
document.getElementById('expand-failures').addEventListener('click',()=>{cards.filter(card=>card.dataset.status==='fail').forEach(card=>{card.classList.remove('hidden');card.open=true;});document.getElementById('task-245cb43_1').scrollIntoView({behavior:'smooth'});});
document.querySelectorAll('a[href^="#task-"]').forEach(link=>link.addEventListener('click',()=>{const target=document.querySelector(link.getAttribute('href'));if(target)target.open=true;}));
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Native Hermes Agent with Claude Opus 5 on 24 AppWorld test_challenge easy tasks in CUGA Eval">
  <title>Native Hermes × AppWorld — CUGA Eval report</title>
  <style>{css}</style>
</head>
<body>
  <header class="hero"><div class="wrap">
    <div class="kicker">Evaluation report · September 23, 2026</div>
    <h1>Native Hermes × AppWorld<br>inside CUGA Eval</h1>
    <p class="hero-copy">A real NousResearch Hermes Agent run on CUGA Eval's 24-task <code>test_challenge_easy</code> subset, using the same model and 25-turn budget as the Harbor comparison. This report records the integration boundary, exact revisions, official verifier results, usage, failures, and redacted task-level evidence.</p>
    <div class="hero-meta"><span>Run: Sep 23, 2026, 21:46:55–22:26:16 CDT</span><span>24 sequential tasks</span><span>1 attempt per task</span><span>Hermes 0.21.4</span></div>
  </div></header>
  <nav><div class="wrap"><a href="#summary">Summary</a><a href="#configuration">Configuration</a><a href="#architecture">Architecture</a>{comparison_nav}<a href="#cost">Usage &amp; cost</a><a href="#failures">Failures</a><a href="#results">All tasks</a><a href="#reproduce">Reproduce</a><a href="#limits">Limits</a></div></nav>
  <main class="wrap">
    <section class="section" id="summary">
      <div class="section-head"><h2>Run summary</h2><p>The official AppWorld evaluator inspected each task's final world state. Binary success requires every task-specific assertion to pass; the mean test pass rate retains partial verifier credit.</p></div>
      <div class="cards">
        {stat('Task goal completion', f'{pass_pct:.2f}%', f'{passed} of {total} tasks')}
        {stat('Mean test pass rate', f'{mean_pct:.2f}%', 'Unweighted mean across tasks')}
        {stat('Verifier assertions', f'{checks_passed}/{checks_total}', 'Passed checks across all tasks')}
        {stat('Runtime exits', str(runtime_failures), 'All at the 25-turn boundary')}
        {stat('Agent execution', fmt_duration(agent_time), 'Sequential subprocess time')}
        {stat('Estimated run cost', f'${run_cost:.2f}', 'Assumed cache rates')}
      </div>
      <div class="result-band" aria-label="{passed} passed and {failed} failed"><div class="ok" style="width:{pass_pct:.4f}%"></div><div class="bad" style="width:{100 - pass_pct:.4f}%"></div></div>
      <div class="legend"><span><i class="dot ok"></i>{passed} passed</span><span><i class="dot bad"></i>{failed} failed</span></div>
      <div class="callout"><strong>Result in one sentence</strong><p>Native Hermes completed 18 of 24 tasks; five of the six failures ended at the 25-turn boundary, including two affected by repeated AppWorld <code>place_order</code> internal errors, while one run finished normally with an incorrect numeric answer.</p></div>
    </section>

    <section class="section" id="configuration">
      <div class="section-head"><h2>What ran</h2><p>This is the real Hermes CLI as an isolated subprocess—not the earlier adapter that only used the Hermes label on CUGA Eval's shared LangChain loop.</p></div>
      <div class="panel"><dl class="config-grid">
        <div><dt>Experiment date</dt><dd>September 23, 2026</dd></div>
        <div><dt>Local run window</dt><dd>21:46:55–22:26:16 CDT</dd></div>
        <div><dt>Experiment output</dt><dd><code>{esc(source_name)}</code></dd></div>
        <div><dt>Harness</dt><dd>CUGA Eval external-agent evaluator</dd></div>
        <div><dt>Agent</dt><dd>NousResearch Hermes Agent 0.21.4</dd></div>
        <div><dt>Agent mode</dt><dd><code>hermes --yolo chat -Q</code></dd></div>
        <div><dt>Model</dt><dd><code>{MODEL}</code></dd></div>
        <div><dt>Provider</dt><dd><code>openai</code> via configured compatible endpoint</dd></div>
        <div><dt>Toolset</dt><dd><code>appworld</code> native HTTP MCP</dd></div>
        <div><dt>MCP output mode</dt><dd><code>content_only</code></dd></div>
        <div><dt>Turn cap / timeout</dt><dd>25 / 1,200 seconds per task</dd></div>
        <div><dt>Memory / checkpoints</dt><dd>Disabled / disabled</dd></div>
        <div><dt>Dataset selection</dt><dd><code>test_challenge_easy</code></dd></div>
        <div><dt>Split / difficulty</dt><dd><code>test_challenge</code> / level 1</dd></div>
        <div><dt>Attempts / retries</dt><dd>1 per task / 0 retries</dd></div>
        <div><dt>Scheduling</dt><dd>Sequential, 24 tasks</dd></div>
      </dl></div>
      <h3>Code provenance</h3>
      <div class="panel"><table class="provenance"><thead><tr><th>Component</th><th>Revision used</th><th>Role</th></tr></thead><tbody>
        <tr><td><strong>CUGA Eval</strong></td><td><code>{CUGA_EVAL_BASE_REVISION}</code> plus the native-Hermes branch changes documented by this report</td><td>Started services, created each task world, launched Hermes, extracted the answer, invoked the official evaluator, and aggregated results.</td></tr>
        <tr><td><strong>Hermes Agent 0.21.4</strong><br><code>NousResearch/hermes-agent</code></td><td><code>{HERMES_REVISION}</code></td><td>Performed autonomous reasoning, native MCP discovery, tool calls, and completion. Installed into a dedicated virtual environment.</td></tr>
        <tr><td><strong>AppWorld</strong><br><code>StonyBrookNLP/appworld</code></td><td><code>{APPWORLD_REVISION}</code></td><td>Provided the task data, remote environment and API servers, all-app MCP surface, task isolation, and official evaluation programs.</td></tr>
      </tbody></table></div>
    </section>

    <section class="section" id="architecture">
      <div class="section-head"><h2>Execution and isolation</h2><p>CUGA Eval owns the task lifecycle and scoring. Hermes owns the agent loop and reaches AppWorld exclusively through its native MCP client.</p></div>
      <div class="architecture">
        <div class="node"><b>CUGA Eval loop</b><small>Sequential task selection, registry reset, result aggregation</small></div><div class="arrow">→</div>
        <div class="node"><b>Isolated AppWorld task</b><small>Remote world context and task-specific state</small></div><div class="arrow">⇄</div>
        <div class="node"><b>Native Hermes subprocess</b><small>Fresh home/workspace; 25 turns; no memory or checkpoints</small></div><div class="arrow">⇄</div>
        <div class="node"><b>All-app AppWorld MCP</b><small>HTTP, <code>content_only</code>; Hermes discovers and calls tools</small></div>
      </div>
      <div class="comparison">
        <article><h4>What matches the Harbor run</h4><p>The actual Hermes CLI runs its own loop, uses the same Opus 5 model identifier and 25-turn cap, discovers AppWorld functions through native MCP, and calls the supervisor completion tool.</p></article>
        <article><h4>What differs from Harbor</h4><p>Harbor scheduled containerized trials with an AppWorld sidecar and two-way concurrency. Here, CUGA Eval runs tasks sequentially against its local AppWorld services and launches a fresh isolated Hermes home/workspace for each task.</p></article>
      </div>
      <div class="callout"><strong>Answer and score path</strong><p>The adapter extracts the exact answer from the nested <code>supervisor__complete_task</code> call when present, falls back to Hermes's final message when necessary, then CUGA Eval submits that answer and runs <code>world.evaluate()</code>. No judge model determines task success.</p></div>
    </section>

{comparison_html}

    <section class="section" id="cost">
      <div class="section-head"><h2>Usage and estimated cost</h2><p>Counts come from exported native Hermes sessions. Dollar values are estimates, not provider billing records.</p></div>
      <div class="cost-grid">
        <div class="panel"><div class="token-bars">
          <div class="token-bar"><div class="labels"><b>Input tokens</b><span>{fmt_int(input_tokens)}</span></div><div class="line"><i style="width:100%"></i></div></div>
          <div class="token-bar"><div class="labels"><b>Cache-read tokens</b><span>{fmt_int(cache_reads)}</span></div><div class="line"><i style="width:{cache_reads / max(input_tokens, 1) * 100:.2f}%"></i></div></div>
          <div class="token-bar"><div class="labels"><b>Cache-write tokens</b><span>{fmt_int(cache_writes)}</span></div><div class="line"><i style="width:{cache_writes / max(input_tokens, 1) * 100:.2f}%"></i></div></div>
          <div class="token-bar"><div class="labels"><b>Output tokens</b><span>{fmt_int(output_tokens)}</span></div><div class="line"><i style="width:{output_tokens / max(input_tokens, 1) * 100:.2f}%"></i></div></div>
        </div></div>
        <div class="panel cost-scenarios">
          <div class="cost-row"><span>24-task scored run<br><small>380 LLM calls</small></span><strong>${run_cost:.4f}</strong></div>
          <div class="cost-row"><span>Four earlier development/smoke runs<br><small>one task each</small></span><strong>${DEVELOPMENT_COST - run_cost:.4f}</strong></div>
          <div class="cost-row"><span>All native-Hermes work in this session<br><small>scored run plus smoke/development</small></span><strong>${DEVELOPMENT_COST:.4f}</strong></div>
        </div>
      </div>
      <p>The adapter estimates Opus 5 usage at $3.80/M uncached input tokens and $19/M output tokens, with assumed 0.1× cache-read and 1.25× cache-write multipliers. The endpoint's actual invoice may differ.</p>
    </section>

    <section class="section" id="failures">
      <div class="section-head"><h2>Failure analysis</h2><p>Six tasks received binary reward 0. Five also returned native Hermes exit code 1 at the configured turn boundary. No authentication, transport, rate-limit, or model-not-found failure appeared in the scored run.</p></div>
      <div class="cards" style="grid-template-columns:repeat(3,1fr);margin-bottom:18px">
        {stat('Order API + turn cap', '2', '0d22252, 245cb43')}
        {stat('Turn cap', '3', '277d81d, 4d12842, 5238afc')}
        {stat('Verifier only', '1', 'ba46d91')}
      </div>
      <div class="failure-grid">{failures_html}</div>
      <p><button type="button" class="filter-btn" id="expand-failures">Expand all failed-task evidence below</button></p>
    </section>

    <section class="section" id="results">
      <div class="section-head"><h2>All 24 tasks</h2><p>Task order follows the committed <code>test_challenge_easy</code> list. Each evidence card includes the exact instruction, final output, official assertions, native-session metrics, redacted tool actions, and local artifact locations.</p></div>
      <div class="filters">
        <button class="filter-btn active" data-filter="all">All <span>{total}</span></button>
        <button class="filter-btn" data-filter="pass">Passed <span>{passed}</span></button>
        <button class="filter-btn" data-filter="fail">Failed <span>{failed}</span></button>
        <input class="search" id="task-search" type="search" placeholder="Search ID, instruction, or output" aria-label="Search tasks">
        <span class="subtle">Showing <b id="visible-count">{total}</b></span>
      </div>
      <div class="table-wrap"><table class="results-table"><thead><tr><th>#</th><th>Task</th><th>Instruction</th><th>Result</th><th>Checks</th><th>Agent time</th><th>Calls</th><th>Input</th><th>Output</th><th>Evidence</th></tr></thead><tbody>{rows}</tbody></table></div>
      <div class="task-list">{cards}</div>
    </section>

    <section class="section" id="reproduce">
      <div class="section-head"><h2>Reproduce the run</h2><p>The installer pins both AppWorld and Hermes. Credentials and the OpenAI-compatible endpoint stay outside the repository.</p></div>
      <div class="steps">
        <article class="step"><h3>Configure the repository</h3><p>Create the normal AppWorld environment files and set the model endpoint credentials in the repository's local <code>.env</code>.</p><pre class="codeblock">OPENAI_API_KEY=...\nOPENAI_BASE_URL=https://your-compatible-endpoint/v1</pre></article>
        <article class="step"><h3>Install pinned AppWorld and Hermes</h3><pre class="codeblock">./setup_appworld.sh\n./benchmarks/appworld/setup_hermes.sh</pre><p>The scripts verify AppWorld <code>{APPWORLD_REVISION[:12]}</code> and Hermes <code>{HERMES_REVISION[:12]}</code>.</p></article>
        <article class="step"><h3>Run one smoke task</h3><pre class="codeblock">./benchmarks/appworld/eval.sh --agent hermes --task e775c78_1</pre></article>
        <article class="step"><h3>Run the 24-task subset</h3><pre class="codeblock">./benchmarks/appworld/eval.sh --agent hermes --eval-key test_challenge_easy</pre><p>Override the model or turn cap with <code>APPWORLD_HERMES_MODEL</code> and <code>APPWORLD_HERMES_MAX_TURNS</code> if intentionally testing another configuration.</p></article>
        <article class="step"><h3>Regenerate this report</h3><pre class="codeblock">HARBOR_SUMMARY=/path/to/hermes-appworld-test-challenge-easy-opus5-20260923/run-summary.json\nuv run python benchmarks/appworld/generate_hermes_report.py \\\n  benchmarks/appworld/experiments/outputs/appworld_hermes_20260923_214655_final_report.json \\\n  benchmarks/appworld/reports/hermes-cuga-eval-test-challenge-easy-opus5-20260923.html \\\n  --harbor-summary "$HARBOR_SUMMARY"</pre><p>The source results and native session artifacts remain gitignored because raw traces include simulated account credentials.</p></article>
      </div>
    </section>

    <section class="section" id="limits">
      <div class="section-head"><h2>Interpretation and limits</h2><p>This report establishes that the native integration executes and can be scored. It is one deterministic task set with one attempt per task, not a broad model ranking.</p></div>
      <div class="limits">
        <article class="limit"><strong>Single attempt</strong><p>No repeated trials or confidence interval were computed. A 75% pass rate describes this run only.</p></article>
        <article class="limit"><strong>Harnesses are not identical</strong><p>The agent/model/turn cap align with Harbor, but process isolation, scheduling, prompts, service topology, and harness completion behavior differ.</p></article>
        <article class="limit"><strong>Estimated cost</strong><p>Token counts are session evidence; dollar totals use assumed rates and cache multipliers rather than billing exports.</p></article>
        <article class="limit"><strong>Turn-cap sensitivity</strong><p>Five failures ended at the 25-turn boundary. Raising the cap may change both success and cost, but this run does not measure that counterfactual.</p></article>
        <article class="limit"><strong>AppWorld endpoint defects</strong><p>Two failures include repeated <code>place_order</code> internal errors and partial database writes. The verifier correctly scored the resulting state as failed.</p></article>
        <article class="limit"><strong>Redacted evidence</strong><p>The committed HTML omits raw MCP result bodies and credential values. Full local artifacts are retained under the gitignored Hermes run directory.</p></article>
      </div>
    </section>

    <footer class="foot"><p><strong>Evidence source:</strong> <code>{esc(source_name)}</code>, produced September 23, 2026. The report embeds aggregate metrics, every official verifier assertion, final outputs, and credential-redacted action traces for all 24 tasks. Raw JSON, stdout, prompts, and exported Hermes sessions remain local and uncommitted.</p></footer>
  </main>
  <script>{script}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CUGA Eval final_report.json")
    parser.add_argument("output", type=Path, help="Destination HTML report")
    parser.add_argument(
        "--harbor-summary",
        type=Path,
        help="Optional Harbor run-summary.json to add an aggregate and task comparison",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if len(data.get("results") or []) != 24:
        raise SystemExit("Expected the 24-task test_challenge_easy result set")
    harbor = None
    harbor_source = None
    if args.harbor_summary:
        harbor = json.loads(args.harbor_summary.read_text(encoding="utf-8"))
        if len(harbor.get("trials") or []) != 24:
            raise SystemExit("Expected a Harbor summary with the same 24-task result set")
        cuga_tasks = {str(result["task_name"]) for result in data["results"]}
        harbor_tasks = {str(trial["task_id"]).replace("-", "_") for trial in harbor["trials"]}
        if harbor_tasks != cuga_tasks:
            raise SystemExit("Harbor and CUGA Eval summaries do not contain the same task IDs")
        harbor_source = f"{args.harbor_summary.parent.name}/{args.harbor_summary.name}"
    rendered = generate(data, str(args.input), harbor, harbor_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} ({len(rendered):,} characters)")


if __name__ == "__main__":
    main()
