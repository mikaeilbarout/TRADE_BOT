@echo off
REM Runs a specific strategy profile (m1/m5/m15) with auto-restart.
REM Usage: run_with_watchdog.bat m1   (or m5 / m15)
REM Defaults to m1 if no argument is given.
REM To run all three strategies at once, run this file three times with a
REM different profile each time (three separate cmd windows, or three
REM separate Tasks/Services).

set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=m1

cd /d "%~dp0\.."

:loop
echo [%date% %time%] Starting bot (profile: %PROFILE%)...
python mt5\live_bot_mt5.py --profile %PROFILE%
echo [%date% %time%] Bot stopped (exit code %errorlevel%) — restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
