#!/usr/bin/env bash
set -euo pipefail

# Install the real NousResearch Hermes Agent in this benchmark's ignored,
# workspace-local directories. No model credentials are read or copied here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_REVISION="d6c9fb8ee8f54bc88897685ba422a60af2ecfab2"
HERMES_INSTALL_DIR="${APPWORLD_HERMES_INSTALL_DIR:-$SCRIPT_DIR/hermes-agent}"
HERMES_DATA_DIR="${APPWORLD_HERMES_HOME:-$SCRIPT_DIR/.hermes}"
INSTALLER_URL="https://raw.githubusercontent.com/NousResearch/hermes-agent/${HERMES_REVISION}/scripts/install.sh"

mkdir -p "$HERMES_DATA_DIR"

echo "Installing Hermes $HERMES_REVISION into $HERMES_INSTALL_DIR..."
curl -fsSL "$INSTALLER_URL" | bash -s -- \
  --dir "$HERMES_INSTALL_DIR" \
  --hermes-home "$HERMES_DATA_DIR" \
  --commit "$HERMES_REVISION" \
  --force-commit \
  --skip-setup \
  --skip-browser \
  --skip-computer-use \
  --no-skills

actual_revision="$(git -C "$HERMES_INSTALL_DIR" rev-parse HEAD)"
if [ "$actual_revision" != "$HERMES_REVISION" ]; then
  echo "Error: expected Hermes $HERMES_REVISION, found $actual_revision" >&2
  exit 1
fi

HERMES_HOME="$HERMES_DATA_DIR" "$HERMES_INSTALL_DIR/venv/bin/hermes" --help >/dev/null
echo "Hermes setup complete at revision $actual_revision."
