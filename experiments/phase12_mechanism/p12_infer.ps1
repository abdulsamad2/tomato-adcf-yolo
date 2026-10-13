$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
Add-Content "$base\logs\p12infer.log" ("[{0}] start" -f (Get-Date -Format 'HH:mm:ss'))
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\p12_infer.py" *>> "$base\logs\p12infer.log"
Add-Content "$base\logs\p12infer.log" ("[{0}] rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\P12INFER_DONE" "done"
