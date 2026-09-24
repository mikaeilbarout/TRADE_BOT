@echo off
REM Stops BOTH live bots (SLP2 and Donchian, incl. their windows and the Donchian watchdog)
REM and starts them again in LIVE mode, so both load the current code and .env settings.
REM Open positions stay at the broker with their SL/TP and are picked up again.
REM Also works when a bot is not running: it is simply started.
chcp 65001 >nul
echo.
echo  WARNING: both bots will (re)start and place REAL orders on the logged-in MT5 account.
echo  Make sure MetaTrader 5 is open, logged in, and Algo Trading is enabled.
echo.
set /p OK=Type YES to restart both bots:
if /I not "%OK%"=="YES" (echo Cancelled. & pause & exit /b)

powershell -NoProfile -Command "$pat='*SLP2.py --live*','*live_bot_mt5.py --profile m15*','*run_with_watchdog.bat m15*'; $n=0; Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='cmd.exe'\" | Where-Object { $c=$_.CommandLine; ($pat | Where-Object { $c -like $_ }) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $n++ }; 'Stopped ' + $n + ' old bot process(es).'"
timeout /t 5 /nobreak >nul

start "SLP2 LIVE" /D "%~dp0combined_projects\SLP2" cmd /k python SLP2.py --live
start "M15 LIVE" /D "%~dp0combined_projects\M15" cmd /k mt5\run_with_watchdog.bat m15
echo Both bots started in new windows. Close a window (or Ctrl+C in it) to stop that bot.
pause
