$base='C:\Users\absamad\cs'
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& ".venv\Scripts\python.exe" "$base\width_gran.py" *>> "$base\logs\wgran.log"
Set-Content "$base\GRAN_DONE" "done"
