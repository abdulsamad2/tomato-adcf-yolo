$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\p16.log"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "=== PHASE 16A: LPWA-50, 5 seeds ==="
foreach($s in 0..4){
  L "start lpwa50 seed$s"
  & $vpy "$base\p16_run.py" --width lpwa50 --seed $s *>> $log
  $rc=$LASTEXITCODE; L "end lpwa50 seed$s rc=$rc"
  if($rc -ne 0){ L "FAILURE rc=$rc - queue stopped"; Set-Content "$base\P16_FAILED" "seed$s rc=$rc"; exit $rc }
}
L "=== training complete; eval ==="
& $vpy "$base\p16_eval.py" "0,1,2,3,4" *>> $log; L "eval rc=$LASTEXITCODE"
Start-Sleep 60
L "=== bench ==="
& $vpy "$base\p16_bench.py" *>> $log; L "bench rc=$LASTEXITCODE"
L "=== stats ==="
& $vpy "$base\p16_stats.py" *>> $log; L "stats rc=$LASTEXITCODE"
Set-Content "$base\P16_DONE" (Get-Date -Format s)
L "=== PHASE 16A COMPLETE ==="
