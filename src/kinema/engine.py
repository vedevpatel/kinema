"""Engine fingerprint: a hash of the code that decides verdicts.

Stored alongside every leaderboard result so a results cache can tell
when it was produced by different check logic and must be re-audited -
bumping a version string by hand is exactly the step that gets forgotten.

The hash covers the code's syntax tree with docstrings removed, so
comments, docstrings and formatting do not trigger a full re-audit (about
two nightly runs); anything that can change a verdict or a message does.
"""

from __future__ import annotations

import ast
import hashlib
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ENGINE_FILES = ("dataset.py", "stats.py", "checks/*.py")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    node.body = body[1:] or [ast.Pass()]
    return tree


def _normalized(source: str) -> str:
    """Syntax tree without docstrings; comments and formatting never reach it."""
    return ast.dump(_strip_docstrings(ast.parse(source)), annotate_fields=False)


@lru_cache(maxsize=1)
def engine_version() -> str:
    h = hashlib.sha1()
    for pattern in _ENGINE_FILES:
        for p in sorted(_ROOT.glob(pattern)):
            h.update(p.relative_to(_ROOT).as_posix().encode())
            h.update(_normalized(p.read_text()).encode())
    return h.hexdigest()[:12]
