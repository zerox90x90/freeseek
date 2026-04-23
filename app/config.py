import os
from pathlib import Path

STATE_DIR = Path(os.path.expanduser("~/.deepseek-proxy"))
STATE_FILE = STATE_DIR / "state.json"
PROFILE_DIR = STATE_DIR / "chromium-profile"
WASM_PATH = Path(__file__).parent / "wasm" / "sha3_wasm_bg.wasm"

BASE_URL = "https://chat.deepseek.com/api/v0"
APP_VERSION = os.environ.get("DS_APP_VERSION", "20241129.1")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY")  # optional gateway auth

STATE_DIR.mkdir(parents=True, exist_ok=True)
