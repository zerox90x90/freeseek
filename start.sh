#!/usr/bin/env bash
# Start the DeepSeek proxy server in the background, then launch Claude Code
# pointed at it via ANTHROPIC_BASE_URL. Kills the server on exit.
#
# Usage: ./start.sh [claude args...]
#   Env overrides:
#     PORT              (default 8765)
#     MODEL             (default deepseek-reasoner; append :search to enable
#                        DeepSeek's upstream web-RAG — usually unwanted since
#                        Claude Code has its own WebSearch/WebFetch tools)
#     FAST_MODEL        (default deepseek-chat)
#     PROXY_API_KEY     (optional; also gates the proxy if set)

set -euo pipefail

INVOKE_DIR="$PWD"
cd "$(dirname "$0")"

PORT="${PORT:-8765}"
MODEL="${MODEL:-deepseek-reasoner}"
FAST_MODEL="${FAST_MODEL:-deepseek-chat}"
LOGFILE="${LOGFILE:-/tmp/deepseek-proxy.log}"
PROXY_API_KEY="${PROXY_API_KEY:-local-dev-key}"

if [ ! -x .venv/bin/python ]; then
  echo "error: .venv missing. Run: python3 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

if [ ! -f "$HOME/.deepseek-proxy/state.json" ]; then
  echo "No saved DeepSeek login. Running login flow (a browser will open)..."
  .venv/bin/python -m probe.login
fi

if command -v claude >/dev/null 2>&1; then
  CLAUDE_BIN="$(command -v claude)"
else
  echo "error: 'claude' CLI not found in PATH. Install via: npm i -g @anthropic-ai/claude-code"
  exit 1
fi

if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use"
  exit 1
fi

echo "Starting proxy on 127.0.0.1:$PORT (logs -> $LOGFILE)..."
PROXY_API_KEY="$PROXY_API_KEY" \
  .venv/bin/python -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$PORT" --log-level warning \
    >"$LOGFILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "Stopping proxy (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for healthz
for _ in $(seq 1 30); do
  if curl -sS -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
    break
  fi
  sleep 0.5
done
if ! curl -sS -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
  echo "error: proxy failed to start. Check $LOGFILE"
  exit 1
fi
echo "Proxy ready."

export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export ANTHROPIC_AUTH_TOKEN="$PROXY_API_KEY"
# Claude Code warns if both AUTH_TOKEN and API_KEY are set — unset API_KEY
# inherited from the parent shell so AUTH_TOKEN wins.
unset ANTHROPIC_API_KEY
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$FAST_MODEL"
# Claude Code uses these to decide behavior; advertise a recent major model
# so feature gates (interleaved thinking, plan mode) light up.
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-$FAST_MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-$MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-$FAST_MODEL}"

echo "Launching Claude Code (model=$MODEL, fast=$FAST_MODEL) in $INVOKE_DIR..."
echo ""
cd "$INVOKE_DIR"
"$CLAUDE_BIN" --dangerously-skip-permissions "$@"
CLAUDE_EXIT=$?
exit $CLAUDE_EXIT
