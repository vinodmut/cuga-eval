"""Native Hermes Agent adapter for AppWorld.

Unlike the other external adapters, this module does not put a Hermes label on
the shared LangChain ReAct loop. It starts the real NousResearch Hermes CLI,
lets Hermes discover AppWorld through its native MCP support, and parses the
exported Hermes session for the answer, tool trace, and usage metrics.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from benchmarks.appworld.agents.base import APPWORLD_AGENT_PROMPT, AppWorldInvokeResult

HERMES_REVISION = "d6c9fb8ee8f54bc88897685ba422a60af2ecfab2"
DEFAULT_HERMES_MODEL = "openai/aws/claude-opus-5"
DEFAULT_HERMES_PROVIDER = "openai"
DEFAULT_HERMES_MAX_TURNS = 25
DEFAULT_APPWORLD_MCP_URL = "http://127.0.0.1:10000/mcp/"
HERMES_APPWORLD_PROMPT = f"""{APPWORLD_AGENT_PROMPT}

E. Native AppWorld completion:

- Use only the tools exposed by the `appworld` MCP server. Do not use host
  terminal, browser, or filesystem tools for AppWorld tasks.
- Act fully autonomously and do not ask the user for confirmation.
- After completing the task, call the AppWorld
  `supervisor__complete_task` API with status `success`. Include the exact
  answer when the task asks a question; omit it when no answer is required.
- After that tool call, make the final response only the minimal answer or a
  short completion confirmation. Do not add analysis or an explanation.
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hermes_binary() -> Path:
    return _project_root() / "benchmarks" / "appworld" / "hermes-agent" / "venv" / "bin" / "hermes"


def _missing_hermes_error(binary: Path) -> RuntimeError:
    return RuntimeError(
        f"Native Hermes is not installed at {binary}. Run ./benchmarks/appworld/setup_hermes.sh first."
    )


def _safe_run_name(thread_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)[:160]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content if part
        )
    return "" if content is None else str(content)


def _load_latest_session(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    sessions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("messages"), list):
            sessions.append(record)
    if not sessions:
        return None
    return max(sessions, key=lambda item: float(item.get("started_at") or 0))


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"raw": arguments}
    return {}


def _session_completion_answer(session: dict[str, Any]) -> str | None:
    """Return the answer Hermes submitted to AppWorld's completion API."""
    for message in reversed(session.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        for raw_call in reversed(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            if function.get("name") != "tool_call":
                continue
            wrapper_args = _parse_arguments(function.get("arguments"))
            for nested_call in reversed(wrapper_args.get("calls") or []):
                if not isinstance(nested_call, dict):
                    continue
                name = str(nested_call.get("name") or "")
                if name.endswith("supervisor__complete_task"):
                    answer = _parse_arguments(nested_call.get("arguments")).get("answer")
                    return "N/A" if answer is None else str(answer)
    return None


def _session_answer(session: dict[str, Any]) -> str:
    completion_answer = _session_completion_answer(session)
    if completion_answer is not None:
        return completion_answer
    for message in reversed(session.get("messages") or []):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = _message_text(message.get("content")).strip()
            if content:
                return content
    return ""


def _session_tool_calls(session: dict[str, Any]) -> list[dict[str, Any]]:
    messages = session.get("messages") or []
    tool_results = {
        message.get("tool_call_id"): _message_text(message.get("content"))
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments = _parse_arguments(function.get("arguments"))
            call_id = raw_call.get("id") or raw_call.get("call_id")
            calls.append(
                {
                    "name": function.get("name", "unknown"),
                    "arguments": arguments,
                    "result": tool_results.get(call_id, ""),
                    "tool_call_id": call_id,
                }
            )
    return calls


def _estimated_opus5_cost(session: dict[str, Any]) -> float:
    """Estimate ETE Opus 5 cost using the rates documented for the Harbor run."""
    input_tokens = int(session.get("input_tokens") or 0)
    output_tokens = int(session.get("output_tokens") or 0)
    cache_reads = int(session.get("cache_read_tokens") or 0)
    cache_writes = int(session.get("cache_write_tokens") or 0)
    return (
        input_tokens * 3.80 + output_tokens * 19.00 + cache_reads * 3.80 * 0.10 + cache_writes * 3.80 * 1.25
    ) / 1_000_000


def _session_metrics(session: dict[str, Any], elapsed: float, model: str) -> dict[str, Any]:
    input_tokens = int(session.get("input_tokens") or 0)
    output_tokens = int(session.get("output_tokens") or 0)
    cache_reads = int(session.get("cache_read_tokens") or 0)
    cache_writes = int(session.get("cache_write_tokens") or 0)
    actual_cost = session.get("actual_cost_usd")
    estimated_cost = session.get("estimated_cost_usd")
    cost_status = str(session.get("cost_status") or "unknown")

    if actual_cost is not None:
        total_cost = float(actual_cost)
        cost_status = "actual"
    elif estimated_cost not in (None, 0, 0.0):
        total_cost = float(estimated_cost)
        cost_status = "estimated_by_hermes"
    elif model.endswith("aws/claude-opus-5"):
        total_cost = _estimated_opus5_cost(session)
        cost_status = "estimated_from_documented_rates"
    else:
        total_cost = 0.0

    return {
        "total_tokens": input_tokens + output_tokens,
        "total_llm_calls": int(session.get("api_call_count") or 0),
        "total_cache_input_tokens": cache_reads,
        "total_cost": total_cost,
        "full_execution_time": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_reads,
        "cache_write_tokens": cache_writes,
        "reasoning_tokens": int(session.get("reasoning_tokens") or 0),
        "cost_status": cost_status,
        "hermes_session_id": session.get("id"),
    }


def _fallback_stdout_answer(stdout: str) -> str:
    clean = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", stdout)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("session_id:")]
    return lines[-1] if lines else ""


class HermesAppWorldAgent:
    """Run a pinned native Hermes CLI session against AppWorld's MCP server."""

    is_native_hermes = True
    provides_final_answer = True

    def __init__(
        self,
        tools: list[Any] | None = None,
        *,
        model: str | None = None,
        max_steps: int | None = None,
        max_turns: int | None = None,
        system_prompt: str = HERMES_APPWORLD_PROMPT,
        hermes_binary: str | Path | None = None,
        mcp_url: str | None = None,
        runs_dir: str | Path | None = None,
        timeout_seconds: float | None = None,
        **_: Any,
    ) -> None:
        # Kept for factory compatibility. Native Hermes discovers tools over MCP.
        del tools, max_steps
        self.model_name = (
            model or os.getenv("APPWORLD_HERMES_MODEL") or os.getenv("MODEL_NAME", DEFAULT_HERMES_MODEL)
        )
        self.provider = os.getenv("APPWORLD_HERMES_PROVIDER", DEFAULT_HERMES_PROVIDER)
        self.max_turns = max_turns or int(
            os.getenv("APPWORLD_HERMES_MAX_TURNS", str(DEFAULT_HERMES_MAX_TURNS))
        )
        self.system_prompt = system_prompt
        self.hermes_binary = Path(
            hermes_binary or os.getenv("APPWORLD_HERMES_BIN") or default_hermes_binary()
        ).expanduser()
        self.mcp_url = mcp_url or os.getenv("APPWORLD_MCP_URL", DEFAULT_APPWORLD_MCP_URL)
        self.runs_dir = Path(
            runs_dir
            or os.getenv("APPWORLD_HERMES_RUNS_DIR")
            or _project_root() / "benchmarks" / "appworld" / ".hermes" / "runs"
        ).expanduser()
        self.timeout_seconds = timeout_seconds or float(os.getenv("APPWORLD_HERMES_TIMEOUT_SECONDS", "1200"))

    def _cli_model(self) -> str:
        prefix = f"{self.provider}/"
        return self.model_name[len(prefix) :] if self.model_name.startswith(prefix) else self.model_name

    def _subprocess_env(self, hermes_home: Path) -> dict[str, str]:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ETE_LITELLM_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or ETE_LITELLM_API_KEY is required for native Hermes")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is required for native Hermes")

        env_names = (
            "PATH",
            "HOME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        )
        env = {name: os.environ[name] for name in env_names if name in os.environ}
        env.update(
            {
                "HERMES_HOME": str(hermes_home),
                "HERMES_IGNORE_RULES": "1",
                "NO_COLOR": "1",
                "OPENAI_API_KEY": api_key,
                "OPENAI_BASE_URL": base_url.rstrip("/"),
                "TERMINAL_ENV": "local",
            }
        )
        return env

    def _write_run_files(self, run_dir: Path, intent: str, user_context: str) -> tuple[Path, Path]:
        hermes_home = run_dir / "home"
        workspace = run_dir / "workspace"
        hermes_home.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        (hermes_home / ".no-bundled-skills").touch()

        config = {
            "model": {"default": self._cli_model(), "provider": self.provider},
            "toolsets": ["appworld"],
            "platform_toolsets": {"cli": ["appworld"]},
            "agent": {"max_turns": self.max_turns},
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "checkpoints": {"enabled": False},
            "mcp_servers": {"appworld": {"url": self.mcp_url}},
        }
        (hermes_home / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        prompt = (
            f"{self.system_prompt}\n\n# USER CONTEXT\n{user_context.strip()}\n\n# TASK\n{intent.strip()}\n"
        )
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        return hermes_home, workspace

    async def _run_process(
        self,
        *args: str,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            str(self.hermes_binary),
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            stdout_bytes, _ = await process.communicate()
            return 124, stdout_bytes.decode("utf-8", errors="replace")
        return process.returncode or 0, stdout_bytes.decode("utf-8", errors="replace")

    async def invoke(
        self,
        *,
        intent: str,
        thread_id: str,
        user_context: str = "",
        track_tool_calls: bool = True,
        config: Optional[dict[str, Any]] = None,
    ) -> AppWorldInvokeResult:
        del config
        if not self.hermes_binary.is_file():
            raise _missing_hermes_error(self.hermes_binary)

        run_dir = self.runs_dir / _safe_run_name(thread_id)
        hermes_home, workspace = self._write_run_files(run_dir, intent, user_context)
        prompt_path = run_dir / "prompt.txt"
        stdout_path = run_dir / "hermes.txt"
        session_path = run_dir / "hermes-session.jsonl"
        env = self._subprocess_env(hermes_home)

        command = (
            "--yolo",
            "chat",
            "--query-file",
            str(prompt_path),
            "-Q",
            "--model",
            self._cli_model(),
            "--provider",
            self.provider,
            "--toolsets",
            "appworld",
            "--max-turns",
            str(self.max_turns),
            "--ignore-rules",
        )
        logger.info(f"Starting native Hermes ({self.model_name}, {self.max_turns} turns, MCP {self.mcp_url})")
        started = time.monotonic()
        return_code, stdout = await self._run_process(
            *command, cwd=workspace, env=env, timeout=self.timeout_seconds
        )
        elapsed = time.monotonic() - started
        stdout_path.write_text(stdout, encoding="utf-8")

        export_code, export_stdout = await self._run_process(
            "sessions",
            "export",
            str(session_path),
            "--source",
            "oneshot",
            cwd=workspace,
            env=env,
            timeout=60,
        )
        session = _load_latest_session(session_path)
        if export_code != 0:
            logger.warning(f"Hermes session export failed with exit code {export_code}")

        answer = _session_answer(session) if session else _fallback_stdout_answer(stdout)
        tool_calls = _session_tool_calls(session) if session and track_tool_calls else []
        metrics = (
            _session_metrics(session, elapsed, self.model_name)
            if session
            else {"full_execution_time": elapsed, "cost_status": "unavailable"}
        )
        react_steps = (
            sum(1 for message in session.get("messages") or [] if message.get("role") == "assistant")
            if session
            else None
        )

        error: str | None = None
        if return_code == 124:
            error = f"Native Hermes timed out after {self.timeout_seconds:g} seconds"
        elif return_code != 0:
            error = f"Native Hermes exited with code {return_code}"
        elif not answer:
            error = "Native Hermes produced no final answer"
        if session is None:
            detail = "Hermes session export was unavailable"
            error = f"{error}; {detail}" if error else detail

        metadata = {
            "hermes_revision": HERMES_REVISION,
            "model": self.model_name,
            "provider": self.provider,
            "max_turns": self.max_turns,
            "mcp_url": self.mcp_url,
            "return_code": return_code,
            "export_return_code": export_code,
            "session_id": metrics.get("hermes_session_id"),
            "metrics": metrics,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        if export_stdout.strip():
            logger.debug(f"Hermes export: {export_stdout.strip().splitlines()[-1]}")
        logger.info(
            f"Native Hermes finished in {elapsed:.1f}s with {len(tool_calls)} tool calls "
            f"and {metrics.get('total_tokens', 0)} tokens"
        )
        return AppWorldInvokeResult(
            answer=answer or "N/A",
            tool_calls=tool_calls,
            react_steps=react_steps,
            error=error,
            metrics=metrics,
            artifacts={
                "run_dir": str(run_dir),
                "stdout": str(stdout_path),
                "session": str(session_path),
            },
        )
