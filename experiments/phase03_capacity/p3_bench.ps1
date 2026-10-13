$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location $repo
Add-Content "$base\logs\p3_bench.log" ("[{0}] start benchmark" -f (Get-Date -Format 'HH:mm:ss'))
& "$repo\.venv\Scripts\python.exe" "$base\p3_bench.py" *>> "$base\logs\p3_bench.log"
Add-Content "$base\logs\p3_bench.log" ("[{0}] rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\BENCH_DONE" "done"
