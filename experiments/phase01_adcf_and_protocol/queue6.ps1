$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue6.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== 960 px baseline at 3 seeds (firms up the resolution / AP_small claim) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-960 --seeds 0 1 2 --device 0 *>> $log
L "960 rc=$LASTEXITCODE"
L "=== collect + export record ==="
& $vpy -m adcf_yolo.collect --out results *>> $log
& $vpy "$base\export_record.py" *>> $log
Set-Content "$base\QUEUE6_DONE" (Get-Date -Format s)
L "=== QUEUE6 COMPLETE ==="
