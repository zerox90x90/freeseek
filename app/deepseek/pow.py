"""PoW solver (ported from xtekky/deepseek4free)."""
import base64
import json
from typing import Any

import numpy as np
import wasmtime

from app.config import WASM_PATH


class _Hasher:
    def __init__(self):
        engine = wasmtime.Engine()
        with open(WASM_PATH, "rb") as f:
            module = wasmtime.Module(engine, f.read())
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]

    def _write(self, text: str) -> tuple[int, int]:
        data = text.encode()
        n = len(data)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, n, 1)
        view = self.memory.data_ptr(self.store)
        for i, b in enumerate(data):
            view[ptr + i] = b
        return ptr, n

    def solve(self, challenge: str, salt: str, difficulty: int, expire_at: int) -> int | None:
        prefix = f"{salt}_{expire_at}_"
        exports = self.instance.exports(self.store)
        retptr = exports["__wbindgen_add_to_stack_pointer"](self.store, -16)
        try:
            cp, cl = self._write(challenge)
            pp, pl = self._write(prefix)
            exports["wasm_solve"](self.store, retptr, cp, cl, pp, pl, float(difficulty))
            view = self.memory.data_ptr(self.store)
            status = int.from_bytes(bytes(view[retptr:retptr + 4]), "little", signed=True)
            if status == 0:
                return None
            value = np.frombuffer(bytes(view[retptr + 8:retptr + 16]), dtype=np.float64)[0]
            return int(value)
        finally:
            exports["__wbindgen_add_to_stack_pointer"](self.store, 16)


_hasher: _Hasher | None = None


def solve_challenge(config: dict[str, Any]) -> str:
    global _hasher
    if _hasher is None:
        _hasher = _Hasher()
    answer = _hasher.solve(
        config["challenge"], config["salt"], config["difficulty"], config["expire_at"]
    )
    result = {
        "algorithm": config["algorithm"],
        "challenge": config["challenge"],
        "salt": config["salt"],
        "answer": answer,
        "signature": config["signature"],
        "target_path": config["target_path"],
    }
    return base64.b64encode(json.dumps(result).encode()).decode()
