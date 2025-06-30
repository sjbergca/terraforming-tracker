@echo off
echo 🔁 Running spreadsheet copy and Git push...

REM Activate virtual environment if needed (optional)
REM call venv\Scripts\activate

python copy_data.py

echo ✅ Done.
pause
