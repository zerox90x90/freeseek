#!/usr/bin/env bash
# Install freeseek dependencies: venv, Python deps, Chromium, Claude Code CLI.
#
# Usage: ./install.sh
#   Env overrides:
#     PYTHON   (default python3)
#     SKIP_CLAUDE=1    skip installing the `claude` CLI via npm
#     SKIP_LAUNCHER=1  skip installing the `freeseek` launcher on $PATH

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Checking Python..."
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.11+ first."
  exit 1
fi
PY_VER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_OK=$("$PYTHON" -c 'import sys; print(1 if sys.version_info >= (3,11) else 0)')
if [ "$PY_OK" != "1" ]; then
  echo "error: Python >= 3.11 required, found $PY_VER"
  exit 1
fi
echo "    python $PY_VER OK"

echo "==> Creating virtualenv (.venv)..."
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
else
  echo "    .venv already exists, reusing"
fi

echo "==> Upgrading pip..."
.venv/bin/pip install --upgrade pip >/dev/null

echo "==> Installing Python dependencies (editable)..."
.venv/bin/pip install -e .

echo "==> Installing Playwright Chromium..."
.venv/bin/playwright install chromium

if [ "${SKIP_LAUNCHER:-0}" != "1" ]; then
  echo "==> Installing 'freeseek' launcher..."
  REPO_DIR="$(pwd)"
  TARGET_DIR=""
  IFS=':' read -r -a _path_parts <<< "$PATH"
  for d in "${_path_parts[@]}"; do
    case "$d" in
      "$HOME"/*) ;;
      *) continue ;;
    esac
    [ -d "$d" ] && [ -w "$d" ] || continue
    TARGET_DIR="$d"
    break
  done
  if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="$HOME/.local/bin"
    mkdir -p "$TARGET_DIR"
    echo "    warn: no writable user dir found on \$PATH; created $TARGET_DIR — add it to your PATH."
  fi
  LAUNCHER="$TARGET_DIR/freeseek"
  cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "$REPO_DIR/start.sh" "\$@"
EOF
  chmod +x "$LAUNCHER"
  echo "    installed: $LAUNCHER"
fi

if [ "${SKIP_CLAUDE:-0}" != "1" ]; then
  echo "==> Checking Claude Code CLI..."
  if command -v claude >/dev/null 2>&1; then
    echo "    claude already installed at $(command -v claude)"
  elif command -v npm >/dev/null 2>&1; then
    echo "    installing @anthropic-ai/claude-code via npm..."
    npm i -g @anthropic-ai/claude-code
  else
    echo "    warn: npm not found; skip Claude Code install."
    echo "          install Node, then: npm i -g @anthropic-ai/claude-code"
  fi
fi

if [ ! -f "$HOME/.deepseek-proxy/state.json" ]; then
  echo "==> Running DeepSeek login (browser will open — login is required)..."
  .venv/bin/python -m probe.login
  if [ ! -f "$HOME/.deepseek-proxy/state.json" ]; then
    echo "error: login did not complete — state.json was not saved."
    echo "       re-run ./install.sh and finish the login in the browser."
    exit 1
  fi
else
  echo "==> DeepSeek login already saved, skipping."
fi

echo ""
if [ "${SKIP_LAUNCHER:-0}" != "1" ]; then
  echo "Done. Start with: freeseek   (or ./start.sh from this directory)"
else
  echo "Done. Start with: ./start.sh"
fi
