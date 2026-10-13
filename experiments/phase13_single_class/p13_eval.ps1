$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\p13_eval.py" *>> "$base\logs\p13eval.log"
Add-Content "$base\logs\p13eval.log" ("[{0}] rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\P13EVAL_DONE" "done"
