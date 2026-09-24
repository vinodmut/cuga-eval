# Quick start: add your agent to the AppWorld comparison

```bash
git clone git@github.com:cuga-project/cuga-eval.git && cd cuga-eval
./setup_appworld.sh
uv sync --all-extras
cp .env.example .env
```

Fill these four in `.env`. Only these two `AGENT_SETTING_CONFIG` values are supported:

```bash
AGENT_SETTING_CONFIG=settings.openai.toml   # or settings.groq.toml
MODEL_NAME=<your-model>
OPENAI_BASE_URL=<your-litellm-or-azure-endpoint>
OPENAI_API_KEY=<key>
```

## 1. Check the harness first

```bash
./benchmarks/appworld/smoke_external.sh --agents stub          # no servers, ~10s
./benchmarks/appworld/eval.sh --agent stub --eval-key test_challenge_easy
```

`stub` is a plain chat model on the shared tool loop, no framework. If `stub` scores
and yours doesn't, the bug is in your adapter.

## 2. Write the adapter

```bash
cp benchmarks/appworld/agents/stub.py benchmarks/appworld/agents/myagent.py
```

Read that file's docstring — it is the full instructions. You replace **one method**,
`_call_llm`. Then register the name in two places:

- `agents/factory.py` — `EXTERNAL_AGENT_NAMES` and a branch in `create_appworld_agent`
- `eval.sh` — `is_external_agent()`

For an in-process adapter, keep these contracts:

| | |
|---|---|
| Tools | from `setup_appworld_tools` (`CombinedToolProvider`). Don't build your own list |
| Prompt | `APPWORLD_AGENT_PROMPT`, as a parameter — don't bake in a different default |
| Return | an `AppWorldInvokeResult`; the evaluator reads `answer` and `tool_calls` |

Pass `invoke_callbacks` through to whatever calls your model. They carry the token
counter, and dropping them silently zeroes your cost column.

A native subprocess agent may need a different tool transport. The real Hermes
adapter is the concrete example: it launches a pinned NousResearch Hermes Agent,
connects it to AppWorld's all-app HTTP MCP server, exports the native Hermes
session, and converts that session into `AppWorldInvokeResult`. It does not use
the shared LangChain ReAct loop.

## 3. Run and compare

```bash
./benchmarks/appworld/eval.sh --agent myagent --eval-key test_challenge_easy
./benchmarks/appworld/compare.sh --eval-key test_challenge_easy --agents cuga,myagent --runs 3
uv run pytest benchmarks/appworld/tests/ -q
```

## 4. Read the numbers honestly

For LangChain adapters, both sides reach tools through `CombinedToolProvider`
against the same registry (`tests/test_tool_provider_parity.py` enforces it).
Three differences remain:

| | CUGA | Yours |
|---|---|---|
| Apps | all; CUGA finds tools itself via `find_tools` | only the task's apps — tool selection is solved for you |
| Prompt | `APPWORLD_SDK_PROMPT` | same base plus filtering/pagination rules CUGA handles in code |
| LLM | CUGA's `LLMManager` | `create_eval_llm`, straight from env |

Both prompts sit side by side in `agents/base.py` so the gap shows in a diff.

`--compare-agents` expands to `cuga,deepagents,openclaw,hermes`. `openclaw` uses
the eval-LLM bridge by default. `hermes` is the actual NousResearch agent and must
be installed separately:

```bash
./benchmarks/appworld/setup_hermes.sh
./benchmarks/appworld/eval.sh --agent hermes --task 82e2fac_1
```

The Hermes run is not strict transport parity with CUGA: both receive all
AppWorld apps, but CUGA uses its registry-backed SDK integration and Hermes uses
AppWorld's native MCP server. The result report records Hermes's native token,
turn, cost, and session-artifact metadata.
