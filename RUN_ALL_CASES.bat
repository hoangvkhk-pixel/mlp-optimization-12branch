@echo off
setlocal
cd /d "%~dp0"
rem Runs every agreed MLP case (paired MTOW, paired q_T, H>=200 sensitivity, extra seeds), 12 threads.
rem Optional: set ALL_SEEDS=<n> (default 10) before calling. Safe to call again after an interruption.
call "%~dp0SETUP_ENV.bat"
if errorlevel 1 goto failed
if not defined ALL_SEEDS set "ALL_SEEDS=10"
"%PYTHON_EXE%" "%~dp0run_all_cases.py" --seeds %ALL_SEEDS%
if errorlevel 1 goto failed
echo ALL CASES DONE
if not defined NO_PAUSE pause
exit /b 0
:failed
echo FAILED. See messages above. Call this file again to continue.
if not defined NO_PAUSE pause
exit /b 1
