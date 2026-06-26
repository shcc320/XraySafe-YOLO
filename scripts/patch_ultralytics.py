from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path


MARKER = "# XraySafe-YOLO custom-module patch"
IMPORT_LINE = "from xraysafe_yolo.modules import CGOA, LSAFF  # XraySafe-YOLO custom-module patch"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def locate_tasks_py() -> Path:
    spec = importlib.util.find_spec("ultralytics.nn.tasks")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate ultralytics.nn.tasks. Install ultralytics first.")
    return Path(spec.origin)


def patch_tasks_source(tasks_py: Path | None = None) -> Path:
    """Patch Ultralytics parse_model so YAML files can use CGOA and LSAFF.

    Ultralytics resolves YAML module names through globals() inside
    ultralytics.nn.tasks.parse_model.  This source-level patch adds our module
    import, registers CGOA as a normal single-input block, and adds a small
    branch for the multi-input LSAFF block.  The original file is backed up as
    tasks.py.xraysafe.bak the first time the patch is applied.
    """
    ensure_repo_on_path()
    tasks_py = locate_tasks_py() if tasks_py is None else Path(tasks_py)
    text = tasks_py.read_text(encoding="utf-8")
    original = text

    if IMPORT_LINE not in text:
        # Insert after the torch import if possible; otherwise prepend after future imports.
        if "import torch\n" in text:
            text = text.replace("import torch\n", "import torch\n" + IMPORT_LINE + "\n", 1)
        else:
            text = IMPORT_LINE + "\n" + text

    # Register CGOA as a base module so parse_model passes (c1, c2, ...).
    if "\n            CGOA," not in text and "\n        CGOA," not in text:
        pattern = r"(base_modules\s*=\s*frozenset\(\s*\{\s*\n)"
        text, n = re.subn(pattern, r"\1            CGOA,\n", text, count=1, flags=re.S)
        if n == 0:
            raise RuntimeError("Could not find base_modules set in ultralytics.nn.tasks.parse_model")

    # Add LSAFF branch before the Detect branch.  LSAFF is multi-input and
    # single-output; it needs the list of input channel counts.
    if "elif m is LSAFF:" not in text:
        lsaff_branch = """
        elif m is LSAFF:
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[x] for x in f], c2, *args[1:]]
"""
        detect_patterns = [
            # Ultralytics 8.4.x / YOLO11: "elif m in frozenset(\n    {\n        Detect, ..."
            r"\n\s{8}elif m in frozenset\(\s*\{\s*Detect,",
            # Older variants occasionally use plain set/tuple syntax.
            r"\n\s{8}elif m in \{\s*Detect,",
            r"\n\s{8}elif m in \(\s*Detect,",
        ]
        inserted = False
        for pat in detect_patterns:
            m = re.search(pat, text)
            if m:
                text = text[: m.start()] + lsaff_branch + text[m.start():]
                inserted = True
                break
        if not inserted:
            raise RuntimeError("Could not locate Detect branch in ultralytics.nn.tasks.parse_model")

    if text != original:
        backup = tasks_py.with_suffix(tasks_py.suffix + ".xraysafe.bak")
        if not backup.exists():
            shutil.copy2(tasks_py, backup)
        tasks_py.write_text(text, encoding="utf-8")
    return tasks_py


def register_runtime_globals() -> None:
    """Also register modules in the already-imported tasks module, if present."""
    ensure_repo_on_path()
    try:
        import ultralytics.nn.tasks as tasks
        from xraysafe_yolo.modules import CGOA, LSAFF

        tasks.CGOA = CGOA
        tasks.LSAFF = LSAFF
    except Exception:
        # The source patch is the critical step; runtime registration is best-effort.
        pass


def ensure_ultralytics_patched() -> None:
    patch_tasks_source()
    register_runtime_globals()


if __name__ == "__main__":
    p = patch_tasks_source()
    register_runtime_globals()
    print(f"Ultralytics patched for XraySafe-YOLO custom modules: {p}")
