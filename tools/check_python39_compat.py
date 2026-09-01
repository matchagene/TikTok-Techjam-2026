"""Static Python 3.9 compatibility checks for project source files."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__"}


def python_files(root: Path = ROOT) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _contains_bit_or(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
        for child in ast.walk(node)
    )


def check_file(path: Path) -> List[str]:
    errors: List[str] = []
    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:
        return [f"{path}: Python 3.9 syntax error at line {exc.lineno}: {exc.msg}"]

    has_future = _has_future_annotations(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name == "TypeAlias":
                    errors.append(
                        f"{path}:{node.lineno}: typing.TypeAlias is unavailable on target Python 3.9"
                    )

        # PEP 604 union annotations (X | Y) are safe on Python 3.9 only when
        # annotation evaluation is postponed. Without the future import they
        # are evaluated at function/class definition time and can fail.
        annotation = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
        elif isinstance(node, ast.AnnAssign):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        if annotation is not None and _contains_bit_or(annotation) and not has_future:
            errors.append(
                f"{path}:{getattr(node, 'lineno', '?')}: PEP 604 annotation requires "
                "'from __future__ import annotations' on Python 3.9"
            )

    # Runtime type aliases are still evaluated even with postponed annotations.
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is not None and _contains_bit_or(value):
            errors.append(
                f"{path}:{node.lineno}: module-level runtime 'X | Y' alias is not Python 3.9-safe"
            )

    return errors


def main() -> int:
    errors: List[str] = []
    files = list(python_files())
    for path in files:
        errors.extend(check_file(path))

    if errors:
        print("Python 3.9 compatibility check FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Python 3.9 compatibility check passed for {len(files)} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
