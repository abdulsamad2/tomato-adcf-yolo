$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\test_eval.log"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location $repo
Add-Content $log ("[{0}] start single test evaluation of the 15 frozen ~18k checkpoints" -f (Get-Date -Format 'MM-dd HH:mm:ss'))
& "$repo\.venv\Scripts\python.exe" "$base\p2b_test_eval.py" *>> $log
Add-Content $log ("[{0}] rc={1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\TESTEVAL_DONE" "done"
