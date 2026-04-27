# freeclaude

Use **GLM (Z.AI)** as the backend for **[Claude Code](https://claude.ai/code)** — free, no API key.

A local proxy that wraps `chat.z.ai` behind an Anthropic-compatible API, so Claude Code can talk to it like it's Claude.

> **Recommendation: use GLM only.** GLM (Z.AI) is the supported, recommended backend. A DeepSeek backend exists for experimentation but is **not recommended** — slower, less reliable tool calling, flaky file uploads, and no server-side session continuation.

![Screenshot](./Screenshot%202026-04-24%20at%202.18.46%E2%80%AFPM.png)

> Only tested on **macOS**.

## Prerequisites

You must have **[Claude Code](https://claude.ai/code)** installed, which requires **Node.js / npm**.

1. Install Node.js (ships with `npm`) — [nodejs.org](https://nodejs.org/) or via Homebrew:

   ```bash
   brew install node
   ```

2. Install Claude Code:

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

(`install.sh` will run step 2 for you automatically if `claude` is not on your `$PATH` — skip with `SKIP_CLAUDE=1`. Node/npm must already be installed.)

## Install

```bash
git clone https://github.com/zerox90x90/freeseek.git
cd freeseek
./install.sh
```

`install.sh` handles everything:
- creates `.venv` and installs Python deps (requires Python 3.11+)
- installs Playwright's Chromium
- drops a `freeseek` launcher into the first writable `$HOME/*` directory on your `$PATH` (skip with `SKIP_LAUNCHER=1`)
- installs the Claude Code CLI via `npm` if missing (skip with `SKIP_CLAUDE=1`)
- runs the one-time Z.AI browser login (required — complete it to finish install)

## Run

Backend is **GLM (Z.AI)** — the only recommended path.

```bash
freeseek                              # GLM
MODEL=glm-5.1:search freeseek         # pick a GLM model
```

Or from the repo directly:

```bash
./start.sh             # GLM
```

### DeepSeek (not recommended)

Kept for experimentation only. Expect rough edges.

```bash
freeseek deepseek
MODEL=deepseek-reasoner freeseek deepseek
./start.sh deepseek
```

First run on the DeepSeek backend triggers a one-time browser login for `chat.deepseek.com`.

## License

MIT
