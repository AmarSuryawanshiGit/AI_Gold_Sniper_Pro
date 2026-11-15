@echo off
call .venv\Scripts\activate.bat
python -m PyInstaller --noconfirm --clean --windowed --onedir --name AI_Gold_Sniper_Pro --collect-all PySide6 --collect-all matplotlib --hidden-import MetaTrader5 main.py
if errorlevel 1 (echo Build failed & pause & exit /b 1)
echo EXE: dist\AI_Gold_Sniper_Pro\AI_Gold_Sniper_Pro.exe
pause
