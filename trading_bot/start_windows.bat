@echo off
REM Double-click to run the trading alert bot on Windows. Close this window to stop it.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
title Crypto trading bot

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python is not installed.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist .venv (
  echo First run: setting up, this takes a few minutes...
  %PY% -m venv .venv || (pause & exit /b 1)
)
.venv\Scripts\python -m pip install -q -r requirements.txt || (pause & exit /b 1)

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Put your Telegram token and chat ID in the file that opens, save it, then double-click this file again.
  notepad .env
  exit /b 0
)

:loop
.venv\Scripts\python run.py
if %errorlevel%==3 (
  notepad .env
  pause
  exit /b 1
)
echo.
echo Bot stopped. Restarting in 30 seconds (close this window to stop for good)...
timeout /t 30 /nobreak >nul
goto loop
