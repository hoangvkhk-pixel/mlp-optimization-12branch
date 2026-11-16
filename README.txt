MLP optimiser - ALL AGREED CASES, strong machine (12 threads), 20261113
========================================================================

One command runs every MLP case needed to rewrite the paper:

    RUN_ALL_CASES.bat            (optional: set ALL_SEEDS=<n> first; default 10 extra seeds per objective)

Cases, in this order (each = the 12 original12 branches, NP = 140, same problem definition as the AVL package):
  1. paired_mtow   init_mtow,  OPT_SEED 20261112, H free 500..10000 m     -> gen_mlp_mtow_np140_s20261112
                   (same initial population and seed as the AVL MTOW run)
  2. paired_qt     init_h500,  OPT_SEED 20261112, H fixed 500 m           -> gen_mlp_qt_np140_s20261112
                   (same initial population and seed as the AVL q_T run)
  3. sens_hmin200  as case 1 but H lower bound 200 m (altitude-bound sensitivity)
                   -> gen_mlp_mtow_hmin200_np140_s20261112, initial population init_mtow_h200 (seed 20260526)
  4. seed k = 1..N, MTOW and q_T, each with its own initial population (seed 20260526+100k) and
     OPT_SEED = 20261112+k                                                -> gen_mlp_<mtow|qt>_np140_s<20261112+k>

The script skips a finished case (all 12 branch_done.json present) and resumes an interrupted one from its
checkpoints. If a case fails it stops (no automatic retry); fix the cause and call RUN_ALL_CASES.bat again.
Every case start/end/duration is appended to all_cases_log.csv. Each output folder has run_manifest.json
(settings, seed, python/numpy/scipy versions, model folders) and per-branch timing.csv, branch_summary.json,
penalty_summary.csv; <outdir>\timing_summary.json gives the totals.
Single cases can still be run with run_mlp_mtow.bat, run_mlp_qt.bat, run_mlp_multiseed.bat.
Only part of the cases: python run_all_cases.py --groups paired   (groups: paired, sens, multi; --seeds N; --first-seed K).

Models (included): models\aero_mlp_normal_macfix_300k and models\aero_mlp_duck_macfix_300k
  trained on 300k AVL labels (25,000 per branch; 20 inputs, 6 outputs cx cy mz_ref mx_beta my_beta K).
  Test R2 >= 0.998 for every output.

Threads: 12 (OMP/OPENBLAS/MKL_NUM_THREADS default 12, OPT_CORES 12). MLP population batching with greenlet is ON
  when the greenlet module is importable (NEW20_POPULATION_BATCH=1): all candidates of a population are evaluated
  cooperatively and their MLP queries go to TensorFlow as one batch per family. Without greenlet the bat files warn
  and fall back to the (about 30x slower) per-candidate evaluation. Batched inference changes floating-point rounding
  slightly, so SHADE trajectories are not bit-identical to the unbatched ones; compare objectives, not bits.
  In a test on normal_2_2 (3 generations) the batched and unbatched runs gave the same feasible counts and best values.

Requirements: 64-bit Python with numpy, pandas, scipy, joblib, openpyxl, scikit-learn, tensorflow (and greenlet for speed).
  Nothing is installed. SETUP_ENV.bat only looks for an installed Python (set PYTHON_EXE to force one,
  AERO_SKIP_CHECK=1 to skip the import check).

Design bounds
-------------
  - sweep of both wings (f_sweep, a_sweep): 0..20 deg.
  - margin = (x_cg - x_np)/MAC: -0.30..-0.05 (static margin >= 5% MAC; negative = stable, checked with AVL).
  - H: MTOW problem 500..10000 m (case 3: 200..10000 m); q_T problem fixed at 500 m.
  - mission 3000 km, max cy 0.6, bias_mz 0.001, mz_omegaz <= -20, epsilon 0.0031, m0 feedback relax 0.5,
    spread stop 0.1.

Resume: a checkpoint is written after every generation (random state included); calling the same .bat/script
again continues from the last finished generation and gives the same result as an uninterrupted run.
