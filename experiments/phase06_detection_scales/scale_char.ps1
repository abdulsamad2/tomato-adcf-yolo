$base='C:\Users\absamad\cs'
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& ".venv\Scripts\python.exe" "$base\scale_char.py" *>> "$base\logs\scale_char.log"
Set-Content "$base\SCALE_DONE" "done"
