$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\p17.log"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "=== PHASE 17: localization KD into 0.75x, 5 seeds, one at a time ==="
foreach($s in 0..4){
  L "start kd075 seed$s"
  & $vpy "$base\p17_run.py" --seed $s *>> $log
  $rc=$LASTEXITCODE; L "end kd075 seed$s rc=$rc"
  if($rc -ne 0){ L "FAILURE rc=$rc - queue stopped"; Set-Content "$base\P17_FAILED" "seed$s rc=$rc"; exit $rc }
}
L "=== PHASE 17 training complete ==="
Set-Content "$base\P17_DONE" (Get-Date -Format s)
