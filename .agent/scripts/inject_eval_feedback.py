import json
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    last_eval_file = repo_root / ".agent" / "state" / "last_eval.json"

    inject_steps = []

    if last_eval_file.exists():
        try:
            with open(last_eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("status") == "error":
                msg = (
                    f"⚠️ [Deterministic Check FAILED] {data.get('message')}\n"
                    f"Archivo: {data.get('file')}\n"
                    f"Código problemático: {data.get('text', '')}\n"
                    f"Corrige la sintaxis antes de continuar o realizar commits."
                )
                inject_steps.append({"ephemeralMessage": msg})

            # Clean up after reading to prevent stale injections
            last_eval_file.unlink(missing_ok=True)
        except Exception:
            pass

    print(json.dumps({"injectSteps": inject_steps}))


if __name__ == "__main__":
    main()
