#!/bin/bash
# Smoke-test in-process external agents against the configured LLM.
#
# Usage:
#   ./smoke_external.sh
#   ./smoke_external.sh --agents stub,deepagents
#   ./smoke_external.sh --native-sdk   # try an OpenClaw native client
#
# Set in .env (repo root):
#   AGENT_SETTING_CONFIG=settings.openai.toml
#   OPENAI_API_KEY=...
#   OPENAI_BASE_URL=https://your-litellm-or-azure-endpoint.example.com
#   MODEL_NAME=your-model-name

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "Smoke test — external agents (no CUGA / no AppWorld)"
echo "  AGENT_SETTING_CONFIG=${AGENT_SETTING_CONFIG:-<unset>}"
echo "  MODEL_NAME=${MODEL_NAME:-<unset>}"
echo "  OPENAI_BASE_URL=${OPENAI_BASE_URL:-${LITE_LLM_URL:-<unset>}}"
echo ""

uv run --no-sync python -m benchmarks.appworld.smoke_external_agents "$@"
