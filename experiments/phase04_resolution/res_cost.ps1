$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location $repo
Add-Content "$base\logs\res_cost.log" ("[{0}] start" -f (Get-Date -Format 'HH:mm:ss'))
& "$repo\.venv\Scripts\python.exe" "$base\res_cost.py" *>> "$base\logs\res_cost.log"
Add-Content "$base\logs\res_cost.log" ("[{0}] rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\COST_DONE" "done"
