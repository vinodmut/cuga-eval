#!/usr/bin/env bash
set -euo pipefail

# One-stop setup for the AppWorld benchmark.
#
# What this script does:
#   1. Clones the upstream AppWorld repo into benchmarks/appworld/appworld
#      (skipped if it's already there).
#   2. Registers it as an editable dependency in pyproject.toml under the
#      `appworld` group (via `uv add --editable ... --group appworld`).
#      pyproject.toml is committed without this entry so that `uv sync`
#      works on fresh checkouts; this script adds it locally.
#   3. Runs `appworld install --repo` and `appworld download data` to set up
#      the repo layout and download the benchmark dataset.
#
# After running this script:
#   - `uv sync --group appworld`  -> installs base deps + appworld
#   - `uv sync`                    -> still works; reconciles to base deps
#                                     (re-run with --group appworld to put
#                                     appworld back).
#
# Re-running this script is safe: existing clone/data are preserved unless
# you opt in to a reinstall when prompted.

APPWORLD_DIR="benchmarks/appworld"
APPWORLD_ENV_FILE="${APPWORLD_DIR}/config/appworld.env"
APPWORLD_REPO_DIR="${APPWORLD_DIR}/appworld"
APPWORLD_DATA_DIR="${APPWORLD_REPO_DIR}/data"
APPWORLD_GIT_URL="https://github.com/StonyBrookNLP/appworld"
APPWORLD_REVISION="42b5bcf3cd334fee33f0c37c02070a9f5807add5"

if [ ! -d "$APPWORLD_DIR" ]; then
  echo "Error: '$APPWORLD_DIR' directory not found."
  echo "Run this script from the repository root."
  exit 1
fi

if [ ! -f "$APPWORLD_ENV_FILE" ]; then
  echo "Error: '$APPWORLD_ENV_FILE' file not found."
  exit 1
fi

set -a
. "$APPWORLD_ENV_FILE"
set +a

# Step 1: clone the upstream repo if missing.
if [ ! -d "$APPWORLD_REPO_DIR" ]; then
  echo "Cloning AppWorld into '$APPWORLD_REPO_DIR'..."
  if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required." >&2
    exit 1
  fi
  # Probe via `git lfs version` rather than `command -v git-lfs` — some
  # package managers install LFS only as a git extension and ship no
  # standalone `git-lfs` binary, which would make `command -v` lie.
  if ! git lfs version >/dev/null 2>&1; then
    echo "Warning: git-lfs not found. AppWorld's data files use LFS; install"
    echo "         it (e.g. 'brew install git-lfs && git lfs install') if the"
    echo "         clone or data download fails."
  fi
  git clone "$APPWORLD_GIT_URL" "$APPWORLD_REPO_DIR"
else
  echo "Found existing AppWorld clone at '$APPWORLD_REPO_DIR'."
fi

# Keep the benchmark runtime identical to the AppWorld revision used by the
# reference Harbor run. Refuse to overwrite tracked work in an existing clone.
if [ "$(git -C "$APPWORLD_REPO_DIR" rev-parse HEAD)" != "$APPWORLD_REVISION" ]; then
  if [ -n "$(git -C "$APPWORLD_REPO_DIR" status --porcelain --untracked-files=no)" ]; then
    echo "Error: AppWorld has tracked local changes; refusing to switch revisions." >&2
    exit 1
  fi
  if ! git -C "$APPWORLD_REPO_DIR" cat-file -e "${APPWORLD_REVISION}^{commit}" 2>/dev/null; then
    echo "Fetching pinned AppWorld revision $APPWORLD_REVISION..."
    git -C "$APPWORLD_REPO_DIR" fetch origin "$APPWORLD_REVISION"
  fi
  git -C "$APPWORLD_REPO_DIR" checkout --detach "$APPWORLD_REVISION"
fi

actual_revision="$(git -C "$APPWORLD_REPO_DIR" rev-parse HEAD)"
if [ "$actual_revision" != "$APPWORLD_REVISION" ]; then
  echo "Error: expected AppWorld $APPWORLD_REVISION, found $actual_revision" >&2
  exit 1
fi
echo "Using pinned AppWorld revision $actual_revision."

# Decide whether to redo the data download.
reinstall_data="yes"
if [ -d "$APPWORLD_DATA_DIR" ]; then
  echo "AppWorld data already exists at '$APPWORLD_DATA_DIR'."
  printf "Re-download data and re-run install? [y/N] "
  read -r answer
  case "$answer" in
    y|Y|yes|YES) reinstall_data="yes" ;;
    *) reinstall_data="no" ;;
  esac
fi

# Step 2: register appworld as an editable dep in the `appworld` group.
# `uv add` is idempotent: re-running updates the entry in place.
#
# Older uv releases require --no-workspace to add AppWorld as a plain editable
# source rather than a workspace member. Newer releases removed that option and
# use the desired path-source behavior by default.
echo "Registering AppWorld as an editable dependency (group: appworld)..."
uv_add_args=(--editable "${APPWORLD_REPO_DIR}[mcp]" --group appworld)
if uv add --help | grep -q -- "--no-workspace"; then
  uv_add_args+=(--no-workspace)
fi
uv add "${uv_add_args[@]}"

if [ "$reinstall_data" = "no" ]; then
  echo "Skipping data download. AppWorld is installed and ready."
  exit 0
fi

# Step 3: set up the repo layout and download data. These commands write into
# the current working directory, so we run them from the clone.
cd "$APPWORLD_REPO_DIR"

echo "Setting up AppWorld repository layout..."
uv run python -m appworld.cli install --repo

echo "Downloading AppWorld data..."
uv run python -m appworld.cli download data

cd - > /dev/null

echo ""
echo "AppWorld setup complete."
echo ""
echo "Usage:"
echo "  uv sync --group appworld   # install/refresh with AppWorld"
echo "  uv sync                    # base deps only (AppWorld will be removed"
echo "                             #   from the venv; re-add with --group)"
echo ""
echo "Note: a bare 'uv sync' (without --group appworld) will silently remove"
echo "      AppWorld from the venv — easy to trip over after a rebase. Use"
echo "      'uv sync --group appworld' to keep it installed."
