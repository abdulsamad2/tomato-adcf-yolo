$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location $repo
Add-Content "$base\logs\p9eval.log" ("[{0}] start" -f (Get-Date -Format 'MM-dd HH:mm:ss'))
& "$repo\.venv\Scripts\python.exe" "$base\p9_eval.py" *>> "$base\logs\p9eval.log"
Add-Content "$base\logs\p9eval.log" ("[{0}] rc={1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\P9EVAL_DONE" "done"
