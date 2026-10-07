import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def get_git_diff(repo_root: Path, file_path: str) -> str:
    try:
        git_dir = repo_root / ".git-agent"
        cmd = [
            "git",
            f"--git-dir={git_dir}",
            f"--work-tree={repo_root}",
            "diff",
            "--",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running git diff: {e}"


def check_syntax(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    ast.parse(content, filename=str(file_path))


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    state_dir = repo_root / ".agent" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    last_eval_file = state_dir / "last_eval.json"

    # Read stdin payload from Antigravity PostToolUse hook
    stdin_data = ""
    try:
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.read()
    except Exception:
        pass

    target_file = None
    if stdin_data:
        try:
            payload = json.loads(stdin_data)
            args = payload.get("toolCall", {}).get("args", {})
            target_file = args.get("TargetFile")
        except Exception:
            pass

    eval_result = {"status": "success", "file": target_file or "unknown"}

    if target_file and os.path.exists(target_file):
        path = Path(target_file)
        if path.suffix == ".py":
            try:
                check_syntax(path)
                diff = get_git_diff(repo_root, str(path))
                eval_result = {
                    "status": "success",
                    "file": str(path),
                    "syntax_valid": True,
                    "diff_lines": len(diff.splitlines()) if diff else 0,
                }
            except SyntaxError as e:
                eval_result = {
                    "status": "error",
                    "file": str(path),
                    "error_type": "SyntaxError",
                    "message": f"SyntaxError in {path.name} at line {e.lineno}, col {e.offset}: {e.msg}",
                    "lineno": e.lineno,
                    "text": e.text.strip() if e.text else "",
                }
            except Exception as e:
                eval_result = {
                    "status": "error",
                    "file": str(path),
                    "error_type": type(e).__name__,
                    "message": str(e),
                }

    try:
        with open(last_eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)
    except Exception:
        pass

    # PostToolUse contract expects an empty JSON object: {}
    print("{}")


if __name__ == "__main__":
    main()
