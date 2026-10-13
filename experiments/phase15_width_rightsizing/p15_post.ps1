$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\p15post.log"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "waiting for P15_DONE"
while(-not (Test-Path "$base\P15_DONE")){
  if(Test-Path "$base\P15_FAILED"){ L "training queue failed - post stops"; exit 1 }
  Start-Sleep 300
}
L "=== eval (80 checkpoints) ==="
& $vpy "$base\p15_eval.py" *>> $log; L "eval rc=$LASTEXITCODE"
Start-Sleep 60
L "=== bench (idle-gated) ==="
& $vpy "$base\p15_bench.py" *>> $log; L "bench rc=$LASTEXITCODE"
L "=== stats ==="
& $vpy "$base\p15_stats.py" *>> $log; L "stats rc=$LASTEXITCODE"
Set-Content "$base\P15POST_DONE" (Get-Date -Format s)
L "=== P15 POST COMPLETE ==="
