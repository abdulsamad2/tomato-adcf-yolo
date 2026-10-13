$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\pxpretrain.log"
& $vpy "$base\px_pretrain_eval.py" *>> $log; Add-Content $log "eval rc=$LASTEXITCODE"
& $vpy "$base\px_pretrain_stats.py" *>> $log; Add-Content $log "stats rc=$LASTEXITCODE"
Set-Content "$base\PXPRETRAIN_DONE" (Get-Date -Format s)
