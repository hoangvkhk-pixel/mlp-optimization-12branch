"""Run every agreed MLP case in sequence (12 threads), resumable.

Cases (each = 12 branches, NP=140, bias_mz 0.001, same problem definition as the AVL package):
  paired_mtow   init_mtow, OPT_SEED 20261112, H free 500..10000   (paired with the AVL run)
  paired_qt     init_h500, OPT_SEED 20261112, H fixed 500          (paired with the AVL run)
  sens_hmin200  as paired_mtow but H lower bound 200 (sensitivity to the altitude bound)
  seed k        k = 1..N, own initial population (seed 20260526+100k) and OPT_SEED 20261112+k, mtow and qt

A finished case (all 12 branch_done.json present) is skipped; an interrupted case is continued from its
checkpoints by calling the same .bat with the same OUTDIR. A failing case stops the whole script
(no automatic retry). Every case start/end is appended to all_cases_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BRANCHES = ["normal_1_1", "normal_1_2", "normal_1_3", "normal_2_1", "normal_2_2", "normal_2_3",
            "normal_3_1", "normal_3_2", "normal_3_3", "duck_1_x", "duck_2_x", "duck_3_x"]
BASE_OPT_SEED = 20261112
BASE_INIT_SEED = 20260526


def cases(n_seeds: int, first_seed: int, groups: set[str]):
    out = []
    if "paired" in groups:
        out.append(dict(name="paired_mtow", obj="mtow", init="init_mtow", init_seed=None, opt_seed=BASE_OPT_SEED,
                        outdir=f"gen_mlp_mtow_np140_s{BASE_OPT_SEED}", env={}))
        out.append(dict(name="paired_qt", obj="qt", init="init_h500", init_seed=None, opt_seed=BASE_OPT_SEED,
                        outdir=f"gen_mlp_qt_np140_s{BASE_OPT_SEED}", env={}))
    if "sens" in groups:
        out.append(dict(name="sens_hmin200", obj="mtow", init="init_mtow_h200", init_seed=BASE_INIT_SEED,
                        opt_seed=BASE_OPT_SEED, outdir=f"gen_mlp_mtow_hmin200_np140_s{BASE_OPT_SEED}",
                        env={"NEW20_H_MIN": "200"}))
    if "multi" in groups:
        for k in range(first_seed, first_seed + n_seeds):
            for obj in ("mtow", "qt"):
                seed = BASE_OPT_SEED + k
                out.append(dict(name=f"seed{k}_{obj}", obj=obj, init=f"init_{obj}_{k}",
                                init_seed=BASE_INIT_SEED + 100 * k, opt_seed=seed,
                                outdir=f"gen_mlp_{obj}_np140_s{seed}", env={}))
    return out


def is_done(outdir: Path) -> bool:
    return all((outdir / b / "branch_done.json").exists() for b in BRANCHES)


def case_env(case: dict) -> dict:
    env = dict(os.environ)
    for k in ("NEW20_H_MIN", "NEW20_FIXED_H", "ONLY_BRANCHES_UNUSED"):
        env.pop(k, None)
    if case["obj"] == "qt":
        env["NEW20_FIXED_H"] = "500"
    env.update(case["env"])
    env.update({"NO_PAUSE": "1", "OPT_CORES": "12", "OPT_SEED": str(case["opt_seed"]),
                "BRANCH_INIT_DIR": case["init"], "OUTDIR": case["outdir"],
                "PYTHON_EXE": sys.executable})
    return env


def log_row(row: list) -> None:
    path = ROOT / "all_cases_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["case", "outdir", "started", "ended", "seconds", "status"])
        w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10, help="number of extra seeds per objective (default 10)")
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--groups", default="paired,sens,multi", help="comma list of: paired, sens, multi")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    groups = {g.strip() for g in a.groups.split(",") if g.strip()}
    todo = cases(a.seeds, a.first_seed, groups)
    print(f"{len(todo)} cases: " + ", ".join(c["name"] for c in todo), flush=True)
    for i, c in enumerate(todo, 1):
        outdir = ROOT / c["outdir"]
        state = "done" if is_done(outdir) else ("partial" if outdir.exists() else "new")
        print(f"\n=== [{i}/{len(todo)}] {c['name']}  out={c['outdir']}  init={c['init']}  seed={c['opt_seed']}  state={state}", flush=True)
        if state == "done" or a.dry_run:
            continue
        env = case_env(c)
        init_dir = ROOT / c["init"]
        if not (init_dir / "manifest.csv").exists():
            if c["init_seed"] is None:
                print(f"Missing initial population {init_dir}", flush=True)
                return 1
            r = subprocess.run([sys.executable, str(ROOT / "src" / "generate_branch_initial_populations_v2.py"),
                                "--n-per-branch", "140", "--seed", str(c["init_seed"]), "--outdir", str(init_dir)],
                               cwd=ROOT, env=env)
            if r.returncode != 0:
                print("Initial population generation failed", flush=True)
                return 1
        bat = ROOT / ("run_mlp_mtow.bat" if c["obj"] == "mtow" else "run_mlp_qt.bat")
        t0 = time.time()
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        r = subprocess.run(["cmd", "/c", str(bat)], cwd=ROOT, env=env)
        ok = r.returncode == 0 and is_done(outdir)
        log_row([c["name"], c["outdir"], started, time.strftime("%Y-%m-%d %H:%M:%S"),
                 round(time.time() - t0, 1), "ok" if ok else f"FAILED rc={r.returncode}"])
        if not ok:
            print(f"Case {c['name']} did not finish. Fix the cause, then run this script again: "
                  "finished cases are skipped and this one resumes from its checkpoints.", flush=True)
            return 1
    print("\nALL CASES DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
