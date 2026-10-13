$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\p17post.log"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "waiting for P17_DONE"
while(-not (Test-Path "$base\P17_DONE")){
  if(Test-Path "$base\P17_FAILED"){ L "training queue failed - post stops"; exit 1 }
  Start-Sleep 300
}
L "=== eval (20 checkpoints) ==="
& $vpy "$base\p17_eval.py" *>> $log; $rc=$LASTEXITCODE; L "eval rc=$rc"
if($rc -ne 0){ Set-Content "$base\P17POST_FAILED" "eval rc=$rc"; exit $rc }
Start-Sleep 60
L "=== bench (idle-gated) ==="
& $vpy "$base\p17_bench.py" *>> $log; L "bench rc=$LASTEXITCODE"
L "=== stats ==="
& $vpy "$base\p17_stats.py" *>> $log; L "stats rc=$LASTEXITCODE"
Set-Content "$base\P17POST_DONE" (Get-Date -Format s)
L "=== P17 POST COMPLETE ==="
