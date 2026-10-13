$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\bcd_smoke.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== B/C/D smoke tests: real frozen config, truncated ==="
L "smoke B (25 epochs, covers warmup end at 23.92 ep)"
& $vpy "$base\bcd_run.py" --arm B --seed 0 --smoke 25 *>> $log
L "B rc=$LASTEXITCODE"
L "smoke C (4 epochs, covers warmup end at 3 ep)"
& $vpy "$base\bcd_run.py" --arm C --seed 0 --smoke 4 *>> $log
L "C rc=$LASTEXITCODE"
L "smoke D (4 epochs)"
& $vpy "$base\bcd_run.py" --arm D --seed 0 --smoke 4 *>> $log
L "D rc=$LASTEXITCODE"
Set-Content "$base\BCD_SMOKE_DONE" "done"
L "SMOKEDONE"
