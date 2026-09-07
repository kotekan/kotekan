#!/usr/bin/env python3
"""Execute a .ipynb top to bottom and write the outputs back into it -- no jupyter needed.

    nb_run.py notebook.ipynb [--cwd DIR]

Code cells run in one shared namespace; stdout, stderr, the last expression's repr and any
matplotlib figures left open at the end of a cell become that cell's outputs (text/plain,
image/png). The first failing cell stops the run, its traceback is stored, and the exit code
is 1. This exists so the tutorial notebooks under scripts/gnss can be validated and shipped
with real outputs from an environment that has numpy/matplotlib but not nbformat.
"""
import argparse
import ast
import base64
import io
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
import json


def run_cell(src, ns):
    """Exec `src` in ns like IPython: statements, then echo the trailing expression."""
    tree = ast.parse(src)
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), ns)
    if tail is not None:
        return eval(compile(tail, "<cell>", "eval"), ns)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--cwd", default=None)
    a = ap.parse_args()
    nb = json.load(open(a.notebook))
    if a.cwd:
        os.chdir(a.cwd)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = {"__name__": "__main__"}
    count, failed = 0, False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        outs = []
        count += 1
        cell["execution_count"] = count
        if failed:
            cell["outputs"] = []
            continue
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                val = run_cell(src, ns)
        except Exception:
            val = None
            tb = traceback.format_exc()
            outs.append({"output_type": "error", "ename": "Error", "evalue": "",
                         "traceback": tb.splitlines()})
            failed = True
            print("cell %d FAILED:\n%s" % (count, tb), file=sys.stderr)
        if out.getvalue():
            outs.insert(0, {"output_type": "stream", "name": "stdout",
                            "text": out.getvalue().splitlines(True)})
        if err.getvalue():
            outs.insert(len(outs) - (1 if failed else 0),
                        {"output_type": "stream", "name": "stderr",
                         "text": err.getvalue().splitlines(True)})
        if val is not None:
            outs.append({"output_type": "execute_result", "execution_count": count,
                         "metadata": {}, "data": {"text/plain": [repr(val)]}})
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=90, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(buf.getvalue()).decode()}})
        plt.close("all")
        cell["outputs"] = outs
    json.dump(nb, open(a.notebook, "w"), indent=1)
    print("%s: %d code cells, %s" % (a.notebook, count, "FAILED" if failed else "ok"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
