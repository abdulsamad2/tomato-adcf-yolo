$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== stage 1: 5-fold CV (baseline vs final) ==="
& $vpy -m adcf_yolo.experiments --cv --device 0 *>> $log
L "stage1 rc=$LASTEXITCODE"
L "=== stage 2: noaug pair ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-noaug yolo11s-adcf-noaug --device 0 *>> $log
L "stage2 rc=$LASTEXITCODE"
L "=== stage 3: baseline suite (yolov8s, yolo26s) ==="
& $vpy -m adcf_yolo.experiments --only yolov8s yolo26s --device 0 *>> $log
L "stage3 rc=$LASTEXITCODE"
L "=== stage 4: collect + bench ==="
& $vpy -m adcf_yolo.collect --out results *>> $log
& $vpy -m adcf_yolo.eval.efficiency --device cuda:0 *>> "$base\logs\bench2.log"
L "stage4 rc=$LASTEXITCODE"
Set-Content "$base\QUEUE_DONE" (Get-Date -Format s)
L "=== QUEUE COMPLETE ==="
