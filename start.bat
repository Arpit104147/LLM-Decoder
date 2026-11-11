@echo off
REM Double-click to start LLM Decoder (opens in your browser)
cd /d "%~dp0"
python -c "import gguf, flask, xlsxwriter, openpyxl" 2>nul || pip install -r requirements.txt
python app.py
pause
