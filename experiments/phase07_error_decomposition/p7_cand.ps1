$base='C:\Users\absamad\cs'
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& ".venv\Scripts\python.exe" "$base\p7_cand.py" *>> "$base\logs\p7cand.log"
Set-Content "$base\P7CAND_DONE" "done"
