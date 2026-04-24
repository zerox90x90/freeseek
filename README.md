# freeclaude

Use **DeepSeek** as the backend for **[Claude Code](https://claude.ai/code)** — free, no API key.

A local proxy that wraps `chat.deepseek.com` (the free web UI) behind an Anthropic-compatible API, so Claude Code can talk to it like it's Claude.

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
- runs the one-time DeepSeek browser login (required — complete it to finish install)

## Run

From anywhere:

```bash
freeseek                              # starts proxy + launches Claude Code
MODEL=deepseek-reasoner freeseek      # use the reasoning model
```

Or from the repo directly:

```bash
./start.sh
```

## License

MIT
