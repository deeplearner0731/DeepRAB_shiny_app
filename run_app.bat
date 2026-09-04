@echo off
REM ---------------------------------------------------------------------------
REM Launch the DeepRAB subgroup explorer locally.
REM Double-click this file, or run it from cmd / PowerShell.
REM
REM Interpreter resolution: %PYEXE% if set, otherwise the first candidate below
REM that actually has shiny and tf-keras installed.  Probing for the packages
REM rather than just for the executable matters -- a half-finished
REM `pip install tensorflow` leaves a venv that looks usable and is not.
REM   .venv\Scripts\python.exe     -- the venv described in README.md
REM   C:\dnnenv\Scripts\python.exe -- legacy location
REM   python on PATH
REM
REM TF_USE_LEGACY_KERAS must be set BEFORE TensorFlow loads: TF >= 2.16 ships
REM Keras 3, which removed keras.backend.in_train_phase -- the call that
REM switches the concrete selection layer between its stochastic training
REM branch and its deterministic argmax inference branch.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

if "%PORT%"=="" set "PORT=8000"

if not "%PYEXE%"=="" goto :found

call :try "%~dp0.venv\Scripts\python.exe" && goto :found
call :try "C:\dnnenv\Scripts\python.exe"   && goto :found
call :try "python"                         && goto :found

echo.
echo ERROR: no Python environment with shiny and tf-keras installed.
echo Create one with:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Or point PYEXE at an existing interpreter:
echo     set "PYEXE=C:\path\to\python.exe"
echo.
pause
exit /b 1

REM Probe one candidate.  find_spec does not import TensorFlow, so this costs
REM ~0.2 s rather than ~10 s.  Sets PYEXE and returns 0 on success.
:try
"%~1" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('shiny') and u.find_spec('tf_keras') else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "PYEXE=%~1"
exit /b 0

:found
set TF_USE_LEGACY_KERAS=1
set TF_CPP_MIN_LOG_LEVEL=3

echo.
echo Starting DeepRAB subgroup explorer...
echo Interpreter: %PYEXE%
echo Open this in your browser:  http://127.0.0.1:%PORT%
echo Press Ctrl+C in this window to stop.
echo.
echo (First run takes ~10 extra seconds while TensorFlow loads.)
echo.

start "" "http://127.0.0.1:%PORT%"
"%PYEXE%" -m shiny run --port %PORT% --host 127.0.0.1 app.py

echo.
echo Server stopped.
pause
