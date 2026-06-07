#!/usr/bin/env python
"""Check current-machine readiness for LMOL commands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from typing import Any


CORE_MODULES = ["numpy", "PIL", "yaml"]
RUNTIME_MODULES = ["torch", "transformers", "peft", "bitsandbytes"]


def module_available(name: str) -> bool:
    """Return whether a Python module can be imported."""

    return importlib.util.find_spec(name) is not None


def runtime_import_checks() -> dict[str, Any]:
    """Import runtime symbols that are required by model loading."""

    checks: dict[str, Any] = {}
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        checks["transformers_lmol_symbols"] = bool(AutoProcessor and LlavaForConditionalGeneration)
    except Exception as exc:
        checks["transformers_lmol_symbols"] = False
        checks["transformers_error"] = repr(exc)
    try:
        from peft import LoraConfig, get_peft_model
        checks["peft_lora_symbols"] = bool(LoraConfig and get_peft_model)
    except Exception as exc:
        checks["peft_lora_symbols"] = False
        checks["peft_error"] = repr(exc)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-runtime", action="store_true", help="Fail when model-training dependencies are missing.")
    args = parser.parse_args()
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {},
        "runtime_imports": {},
        "cuda": None,
    }
    modules: dict[str, bool] = {}
    for name in CORE_MODULES + RUNTIME_MODULES:
        modules[name] = module_available(name)
    result["modules"] = modules
    runtime_checks = runtime_import_checks()
    result["runtime_imports"] = runtime_checks
    if modules.get("torch"):
        import torch

        result["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    missing_core = [name for name in CORE_MODULES if not modules[name]]
    missing_runtime = [name for name in RUNTIME_MODULES if not modules[name]]
    if missing_core:
        print(f"Missing core modules: {', '.join(missing_core)}", file=sys.stderr)
        return 1
    if args.strict_runtime and missing_runtime:
        print(f"Missing runtime modules: {', '.join(missing_runtime)}", file=sys.stderr)
        return 1
    if args.strict_runtime and not all(value for key, value in runtime_checks.items() if key.endswith("_symbols")):
        print("Runtime symbol import checks failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
