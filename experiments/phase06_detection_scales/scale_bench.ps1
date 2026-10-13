$base='C:\Users\absamad\cs'
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& ".venv\Scripts\python.exe" "$base\scale_bench.py" *>> "$base\logs\scale_bench.log"
& ".venv\Scripts\python.exe" "$base\scale_geo.py" *>> "$base\logs\scale_geo.log"
Set-Content "$base\BENCH2_DONE" "done"
