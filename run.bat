@echo off
REM One command setup + launch (Windows):  run.bat
cd /d %~dp0
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q jupyter nbconvert ipykernel
python scripts\configure.py
if not exist reports mkdir reports
jupyter nbconvert --to notebook --execute CineMind_Launcher.ipynb --output-dir reports --output launcher_executed --ExecutePreprocessor.timeout=-1
echo Done. Executed notebook: reports\launcher_executed.ipynb
