$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
Add-Content "$base\logs\p13.log" ("[{0}] start" -f (Get-Date -Format 'HH:mm:ss'))
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\p13_run.py" *>> "$base\logs\p13.log"
Add-Content "$base\logs\p13.log" ("[{0}] rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\P13_DONE" "done"
