@echo off
REM --------------------------------------------------------------
REM  Remove the local Python virtual environment (.venv)
REM --------------------------------------------------------------

pushd %~dp0

if exist ".venv" (
    echo Removing .venv folder...
    rmdir /s /q .venv
    echo Done. The virtual environment has been deleted.
) else (
    echo No .venv folder found. Nothing to remove.
)

pause
popd 