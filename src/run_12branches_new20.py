from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import shade_new20 as shade
from original12_branching import BRANCHES, BranchSpec
from problem_v2_spec import BOUNDS_V2, INPUT_COLS_V2


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_run_manifest(outdir: Path, args, seed: int) -> None:
    root = PROJECT_ROOT
    files = sorted((root / "src").glob("*.py")) + sorted((root / "avl_optimize_portable" / "src_avl_full").glob("*.py"))
    files.append(root / "avl_optimize_portable" / "src_avl_full" / "avl.exe")
    init_dir = Path(args.init_dir) if Path(args.init_dir).is_absolute() else root / args.init_dir
    inits = sorted(init_dir.glob("*.npy"))
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NEW20_", "SHADE_", "OPT_", "BRANCH_", "AERO_"))}
    import numpy, scipy
    info = {
        "backend": args.backend, "seed": seed, "cores": args.cores, "n_per_branch": args.n_per_branch,
        "epsilon": args.epsilon, "model_dir": args.model_dir if args.backend == "mlp" else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "env": env,
        "source_sha256": {str(p.relative_to(root)): _sha256(p) for p in files if p.is_file()},
        "initial_population_sha256": {p.name: _sha256(p) for p in inits},
    }
    (outdir / "run_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["avl", "mlp"], required=True)
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--model-dir", type=str, default="models/aero_mlp_v2")
    p.add_argument("--cores", type=int, default=int(os.environ.get("OPT_CORES", "10")))
    p.add_argument("--epsilon", type=float, default=float(os.environ.get("BRANCH_EPSILON", "0.0031")))
    p.add_argument("--init-dir", type=str, default="shared_branch_initial_populations_v2")
    p.add_argument("--n-per-branch", type=int, default=140)
    return p.parse_args()


def base_bounds() -> np.ndarray:
    return np.asarray([BOUNDS_V2[c] for c in INPUT_COLS_V2], dtype=float)


def branch_bounds(branch: BranchSpec) -> np.ndarray:
    bounds = base_bounds()
    bounds[INPUT_COLS_V2.index("a_S_rel")] = branch.a_s_range
    bounds[INPUT_COLS_V2.index("scheme_fuse")] = [branch.scheme_fuse, branch.scheme_fuse]
    bounds[INPUT_COLS_V2.index("scheme_vertical")] = [branch.scheme_vertical, branch.scheme_vertical]
    bounds[INPUT_COLS_V2.index("a_dihedral_mag")] = branch.a_dihedral_range
    fixed_h = os.environ.get("NEW20_FIXED_H", "").strip()
    if fixed_h:
        h_val = float(fixed_h)
        bounds[INPUT_COLS_V2.index("H")] = [h_val, h_val]
    h_min = os.environ.get("NEW20_H_MIN", "").strip()
    if h_min and not fixed_h:
        bounds[INPUT_COLS_V2.index("H")][0] = float(h_min)
    return bounds


def selected_branches():
    only = os.environ.get("ONLY_BRANCHES", "").strip()
    if not only:
        return BRANCHES
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    return [b for b in BRANCHES if b.name in wanted]


def main() -> None:
    args = parse_args()
    os.environ["NEW20_BACKEND"] = args.backend
    requested = int(args.cores)
    args.cores = max(1, min(requested, 12, os.cpu_count() or 12))
    if args.cores != requested:
        print(f"Cores limited from {requested} to {args.cores} (at most 12 AVL worker processes).", flush=True)
    outdir = Path(args.outdir or f"gen_{args.backend}_new20_12branches")
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    init_dir = Path(args.init_dir)
    if not init_dir.is_absolute():
        init_dir = PROJECT_ROOT / init_dir

    seed = int(os.environ.get("OPT_SEED", "20261112"))
    manifest = ["branch,backend,initial_population,output_dir"]
    start = time.time()
    if (outdir / "run_manifest.json").exists():
        with (outdir / "resume_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} run restarted (cores={args.cores}, resume={os.environ.get('OPT_RESUME', '1')})\n")
    else:
        write_run_manifest(outdir, args, seed)
    print(f"RUNNING NEW20 12-BRANCH {args.backend.upper()} OPTIMIZATION")
    print(f"Output : {outdir}")
    print(f"Cores  : {args.cores}")
    print(f"Init   : {init_dir}")

    for branch in selected_branches():
        des_var = branch_bounds(branch)
        init_pop = init_dir / f"{branch.name}_initial_population_{args.n_per_branch}.npy"
        os.environ["SHADE_INIT_POP_OVERRIDE"] = str(args.n_per_branch)
        if init_pop.exists():
            os.environ["SHADE_INITIAL_POP"] = str(init_pop)
        else:
            os.environ.pop("SHADE_INITIAL_POP", None)
        branch_out = outdir / branch.name
        print(f"\nBRANCH {branch.name}")
        print(f"Initial population: {os.environ.get('SHADE_INITIAL_POP', '[generated by SHADE]')}")
        print(f"a_dihedral_mag bounds: {branch.a_dihedral_range[0]}..{branch.a_dihedral_range[1]}")
        done_marker = branch_out / "branch_done.json"
        resume = os.environ.get("OPT_RESUME", "1").strip().lower() not in {"0", "false", "no", "off"}
        if resume and done_marker.exists():
            print(f"[SKIP] {branch.name}: already finished ({done_marker})", flush=True)
            manifest.append(f"{branch.name},{args.backend},{os.environ.get('SHADE_INITIAL_POP', '')},{branch_out}")
            continue
        np.random.seed(seed + BRANCHES.index(branch))
        shade.SHADE_algorithm(
            des_var,
            const=2,
            epsilon=args.epsilon,
            gen_path=str(branch_out),
            cores=args.cores,
            model_dir=args.model_dir if args.backend == "mlp" else None,
        )
        (branch_out / "branch_done.json").write_text(json.dumps({"branch": branch.name, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        manifest.append(f"{branch.name},{args.backend},{os.environ.get('SHADE_INITIAL_POP', '')},{branch_out}")

    (outdir / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    summaries = []
    for branch in selected_branches():
        sp = outdir / branch.name / "branch_summary.json"
        if sp.exists():
            summaries.append(json.loads(sp.read_text(encoding="utf-8")))
    (outdir / "timing_summary.json").write_text(json.dumps({
        "branches": summaries, "seconds_this_session": round(time.time() - start, 1),
        "seconds_all_sessions": round(sum(x.get("seconds_total", 0) for x in summaries), 1),
        "evaluations": int(sum(x.get("evaluations", 0) for x in summaries)),
    }, indent=2), encoding="utf-8")
    print(f"TOTAL TIME: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
