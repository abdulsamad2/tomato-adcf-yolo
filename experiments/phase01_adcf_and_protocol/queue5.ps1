$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue5.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== task 1.1: leakage inflation (eval only, no training) ==="
if(-not (Test-Path "$repo\results\leakage_inflation.json")){
  & $vpy "$base\leak_measure.py" *>> $log
  L "1.1 rc=$LASTEXITCODE"
} else { L "1.1 already done" }
L "=== task 0.4: augmentation finding at 5 seeds (adds seeds 3,4) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s yolo11s-noaug --seeds 0 1 2 3 4 --device 0 *>> $log
L "0.4 rc=$LASTEXITCODE"
L "=== task 1.2: P2 head baseline, no ADCF, 3 seeds ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-p2 --device 0 *>> $log
L "1.2 rc=$LASTEXITCODE"
L "=== task 1.3: CBAM control, 3 seeds ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-cbam --device 0 *>> $log
L "1.3 rc=$LASTEXITCODE"
L "=== task 0.3: latency re-benchmark, 1000 iters, 200 warmup (idle card) ==="
& $vpy -m adcf_yolo.eval.efficiency --device cuda:0 --iters 1000 --warmup 200 *> "$base\logs\bench_1000.log"
L "0.3 rc=$LASTEXITCODE"
L "=== collect + export record ==="
& $vpy -m adcf_yolo.collect --out results *>> $log
& $vpy "$base\export_record.py" *>> $log
Set-Content "$base\QUEUE5_DONE" (Get-Date -Format s)
L "=== QUEUE5 COMPLETE ==="
