$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue7.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== nano models for edge deployment: yolo11n, yolov8n, yolo26n at 3 seeds ==="
& $vpy -m adcf_yolo.experiments --only yolo11n yolov8n yolo26n --device 0 *>> $log
L "nano rc=$LASTEXITCODE"
L "=== latency: batch 1 (deployment-realistic) and batch 16 (compute-bound) ==="
& $vpy -m adcf_yolo.eval.efficiency --device cuda:0 --imgsz 640 --batch 1 --iters 1000 --warmup 200 *> "$base\logs\bench_nano_b1.log"
& $vpy -m adcf_yolo.eval.efficiency --device cuda:0 --imgsz 640 --batch 16 --iters 300 --warmup 50 *> "$base\logs\bench_nano_b16.log"
L "bench rc=$LASTEXITCODE"
L "=== collect + export record ==="
& $vpy -m adcf_yolo.collect --out results *>> $log
& $vpy "$base\export_record.py" *>> $log
Set-Content "$base\QUEUE7_DONE" (Get-Date -Format s)
L "=== QUEUE7 COMPLETE ==="
