@echo off
setlocal
cd /d "%~dp0"
call "%~dp0SETUP_ENV.bat"
if errorlevel 1 goto failed

rem ---- problem definition (identical for the AVL and MLP packages) ----
set "NEW20_BACKEND=mlp"
set "NEW20_OBJECTIVE=q_g_per_ton_km"
set "NEW20_MISSION_L_KM=3000"
set "NEW20_MAX_CY=0.6"
set "NEW20_FIXEDPOINT=0"
set "NEW20_PENALIZE_MZ=1"
set "NEW20_BIAS_MZ=0.001"
set "NEW20_ENABLE_M0_FEEDBACK=1"
set "NEW20_M0_SYNC_RELAX=0.5"
set "NEW20_ENABLE_SPREAD_STOP=1"
set "NEW20_SPREAD_STOP_MIN_FEASIBLE_RATIO=0.1"
set "NEW20_PENALIZE_MZ_OMEGAZ=1"
set "NEW20_MIN_MZ_OMEGAZ=-20"
set "BRANCH_EPSILON=0.0031"
set "NEW20_FIXED_H=500"
set "NEW20_MLP_BATCH=1"

rem ---- run controls (set the variable before calling this file to override) ----
if not defined OPT_CORES set "OPT_CORES=12"
if not defined OPT_SEED set "OPT_SEED=20261112"
if not defined SHADE_MAX_GENERATIONS set "SHADE_MAX_GENERATIONS=0"
if not defined BRANCH_N_PER_BRANCH set "BRANCH_N_PER_BRANCH=140"
if not defined ONLY_BRANCHES set "ONLY_BRANCHES="
if not defined BRANCH_INIT_DIR set "BRANCH_INIT_DIR=init_h500"
if not defined OUTDIR set "OUTDIR=gen_mlp_qt_np%BRANCH_N_PER_BRANCH%_s%OPT_SEED%"
set "SHADE_INITIAL_POP="
set "SHADE_RESUME_STATE="
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=12"
if not defined OPENBLAS_NUM_THREADS set "OPENBLAS_NUM_THREADS=12"
if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=12"

if not defined AERO_MODEL_DIR_NORMAL set "AERO_MODEL_DIR_NORMAL=models\aero_mlp_normal_macfix_300k"
if not defined AERO_MODEL_DIR_DUCK set "AERO_MODEL_DIR_DUCK=models\aero_mlp_duck_macfix_300k"
set "NEW20_MODEL_DIR_NORMAL=%AERO_MODEL_DIR_NORMAL%"
set "NEW20_MODEL_DIR_DUCK=%AERO_MODEL_DIR_DUCK%"
if not defined NEW20_POPULATION_BATCH (
  "%PYTHON_EXE%" -c "import greenlet" >nul 2>nul && (set "NEW20_POPULATION_BATCH=1") || (set "NEW20_POPULATION_BATCH=0" & echo [WARN] greenlet not installed - using per-candidate MLP evaluation, much slower)
)
echo MLP population batching: %NEW20_POPULATION_BATCH%
if not exist "%AERO_MODEL_DIR_NORMAL%\aero_mlp_v2_best.keras" (
  echo Missing %AERO_MODEL_DIR_NORMAL%\aero_mlp_v2_best.keras - copy the retrained models first.
  goto failed
)
if not exist "%AERO_MODEL_DIR_DUCK%\aero_mlp_v2_best.keras" (
  echo Missing %AERO_MODEL_DIR_DUCK%\aero_mlp_v2_best.keras - copy the retrained models first.
  goto failed
)

if not exist "%BRANCH_INIT_DIR%\manifest.csv" (
  "%PYTHON_EXE%" src\generate_branch_initial_populations_v2.py --n-per-branch %BRANCH_N_PER_BRANCH% --outdir "%BRANCH_INIT_DIR%"
  if errorlevel 1 goto failed
)

echo [MLP-QT] cores=%OPT_CORES% seed=%OPT_SEED% init=%BRANCH_INIT_DIR% out=%OUTDIR%
"%PYTHON_EXE%" src\run_12branches_new20.py ^
  --backend mlp ^
  --outdir "%OUTDIR%" ^
  --model-dir "%AERO_MODEL_DIR_NORMAL%" ^
  --cores %OPT_CORES% ^
  --init-dir "%BRANCH_INIT_DIR%" ^
  --n-per-branch %BRANCH_N_PER_BRANCH%
if errorlevel 1 goto failed
echo DONE %OUTDIR%
if not defined NO_PAUSE pause
exit /b 0
:failed
echo FAILED. See messages above.
if not defined NO_PAUSE pause
exit /b 1
