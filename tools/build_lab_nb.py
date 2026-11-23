"""Build a lab's solution + student notebooks from one percent-format source, run the solution, save checkpoints.

    python tools/build_lab_nb.py labs/src/lab1-logistic.py [--no-run]
    python tools/build_lab_nb.py --all

Source format (labs/src/<lab>.py)
  # %% [markdown]          a markdown cell; every line starts with "# " (or is "#")
  # %%                     a code cell
  # >>> TODO 1.2: <hint>   start of a TODO block (code cells only)
  #? X = ___               starter line: shown in the student notebook, hidden in the solution
  # <<<                    end of the TODO block

Outputs: labs/<lab>.ipynb (student: no outputs, TODOs blanked) · labs/solution/<lab>.ipynb (executed, with outputs;
its first cell moves the working directory up to labs/)
· labs/checkpoints/<name>.pkl (from save_checkpoint() calls while the solution runs).
VS Code runs a notebook in the notebook's own folder, so student notebooks live in labs/ next to data/ and mlia.py.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
KERNEL = {"name": "python3", "display_name": "Python 3", "language": "python"}
SOLUTION_CD = ('# เฉลย: ย้ายไปทำงานที่โฟลเดอร์ labs/ (ที่มี data/ และ mlia.py)\n'
               'import os, sys\n'
               'if os.path.basename(os.getcwd()) == "solution":\n'
               '    os.chdir("..")\n'
               'sys.path.insert(0, os.getcwd())')
TODO_START = re.compile(r"^(\s*)# >>> (TODO.*)$")


def parse(src: Path) -> list[tuple[str, str]]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(re.sub(r"^# ?", "", line) for line in text.splitlines())
        if text.strip():
            cells.append((kind, text))

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "md", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def render_code(text: str, student: bool) -> str:
    out, in_todo, indent, starters = [], False, "", []
    for line in text.splitlines():
        m = TODO_START.match(line)
        if m:
            in_todo, indent, starters = True, m.group(1), []
            out.append(f"{indent}# {m.group(2)}" if student else f"{indent}# ✔ {m.group(2)}")
            continue
        if in_todo and line.strip() == "# <<<":
            if student:
                out.extend(starters)
            in_todo = False
            continue
        if in_todo:
            s = line.lstrip()
            if s.startswith("#?"):
                body = s[2:]
                starters.append(line[: len(line) - len(s)] + (body[1:] if body.startswith(" ") else body))
            elif not student:
                out.append(line)
            continue
        out.append(line)
    if in_todo:
        raise ValueError("TODO block without closing '# <<<'")
    return "\n".join(out).rstrip()


def build(src: Path, run: bool = True) -> None:
    cells = parse(src)
    nbs = {}
    for student in (False, True):
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = KERNEL
        nb.metadata["language_info"] = {"name": "python"}
        for kind, text in cells:
            if kind == "md":
                nb.cells.append(nbformat.v4.new_markdown_cell(text))
            else:
                nb.cells.append(nbformat.v4.new_code_cell(render_code(text, student)))
        nbs[student] = nb

    sol = LABS / "solution" / f"{src.stem}.ipynb"
    stu = LABS / f"{src.stem}.ipynb"
    sol.parent.mkdir(exist_ok=True)
    nbs[False].cells.insert(0, nbformat.v4.new_code_cell(SOLUTION_CD))
    nbformat.write(nbs[True], stu)
    print("wrote", stu.relative_to(ROOT))

    if run:
        os.environ["MLIA_SAVE_CHECKPOINTS"] = "1"
        os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
        client = NotebookClient(nbs[False], timeout=900, kernel_name="python3", resources={"metadata": {"path": str(sol.parent)}})
        client.execute()
    nbformat.write(nbs[False], sol)
    print("wrote", sol.relative_to(ROOT), "(executed)" if run else "(not run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    srcs = sorted((LABS / "src").glob("lab*.py")) if a.all else a.src
    if not srcs:
        sys.exit("give a source file or --all")
    for s in srcs:
        build(s.resolve(), run=not a.no_run)
    ck = LABS / "checkpoints"
    if ck.exists():
        print("checkpoints:", ", ".join(sorted(p.stem for p in ck.glob("*.pkl"))))


if __name__ == "__main__":
    main()
