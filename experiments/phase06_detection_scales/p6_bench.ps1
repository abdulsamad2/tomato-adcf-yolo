$base='C:\Users\absamad\cs'
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& ".venv\Scripts\python.exe" "$base\p6_bench.py" *>> "$base\logs\p6_bench.log"
Set-Content "$base\P6BENCH_DONE" "done"
