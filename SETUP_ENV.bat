@echo off
rem Uses an already installed 64-bit Python (no pip, no internet). Set PYTHON_EXE to force one.
cd /d "%~dp0"
if /i "%AERO_SKIP_CHECK%"=="1" if not defined PYTHON_EXE set "PYTHON_EXE=python"
if /i "%AERO_SKIP_CHECK%"=="1" (
    if not exist "avl_optimize_portable\src_avl_full\avl.exe" (
        echo Missing avl_optimize_portable\src_avl_full\avl.exe
        exit /b 1
    )
    echo Environment check skipped ^(AERO_SKIP_CHECK=1^). Using: %PYTHON_EXE%
    exit /b 0
)
if not defined AERO_IMPORTS set "AERO_IMPORTS=numpy, pandas, scipy, joblib, openpyxl, sklearn, tensorflow"
set "AERO_CHECK=import sys, %AERO_IMPORTS%; assert sys.maxsize > 2**32"
set "PYTHON_OK="
if defined PYTHON_EXE call :try "%PYTHON_EXE%"
if not defined PYTHON_OK call :try "C:\ProgramData\Miniconda3\envs\vsppytools\python.exe"
if not defined PYTHON_OK call :try python
if not defined PYTHON_OK call :try py
if not defined PYTHON_OK (
    echo No 64-bit Python with %AERO_IMPORTS% was found.
    echo Set it first, for example: set "PYTHON_EXE=C:\path\to\python.exe"
    exit /b 1
)
set "PYTHON_EXE=%PYTHON_OK%"
if not exist "avl_optimize_portable\src_avl_full\avl.exe" (
    echo Missing avl_optimize_portable\src_avl_full\avl.exe
    exit /b 1
)
echo Using Python: %PYTHON_EXE%
exit /b 0

:try
"%~1" -c "%AERO_CHECK%" >nul 2>nul
if not errorlevel 1 set "PYTHON_OK=%~1"
exit /b 0
