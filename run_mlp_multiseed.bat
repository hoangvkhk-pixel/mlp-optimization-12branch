@echo off
rem Extra MLP runs with different initial populations and seeds.
rem Usage: run_mlp_multiseed.bat mtow|qt FIRST LAST   (seed k uses init_<obj>_k and OPT_SEED=20261112+k)
setlocal enabledelayedexpansion
cd /d "%~dp0"
if "%~3"=="" (
  echo Usage: run_mlp_multiseed.bat mtow^|qt FIRST LAST
  exit /b 1
)
set "OBJ=%~1"
set "NO_PAUSE=1"
call "%~dp0SETUP_ENV.bat"
if errorlevel 1 exit /b 1
for /l %%k in (%~2,1,%~3) do (
  set /a "OPT_SEED=20261112+%%k"
  set /a "INIT_SEED=20260526+%%k*100"
  set "BRANCH_INIT_DIR=init_!OBJ!_%%k"
  set "OUTDIR=gen_mlp_!OBJ!_np140_s!OPT_SEED!"
  if not exist "!OUTDIR!\run_manifest.json" (
    if /i "!OBJ!"=="qt" ( set "NEW20_FIXED_H=500" ) else ( set "NEW20_FIXED_H=" )
    if not exist "!BRANCH_INIT_DIR!\manifest.csv" "%PYTHON_EXE%" src\generate_branch_initial_populations_v2.py --n-per-branch 140 --seed !INIT_SEED! --outdir "!BRANCH_INIT_DIR!"
    call "%~dp0run_mlp_!OBJ!.bat"
  ) else (
    echo skip seed %%k: !OUTDIR! already started
  )
)
exit /b 0
