$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"; $env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
Set-Location "$base\tomato-adcf-yolo"
$vpy="$base\tomato-adcf-yolo\.venv\Scripts\python.exe"; $log="$base\logs\p15.log"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "=== PHASE 15 queue start (20 runs, seed-major) ==="
foreach($s in 0..4){
  foreach($w in 'w100','w075','w050','w035'){
    L "start $w seed$s"
    & $vpy "$base\p15_run.py" --width $w --seed $s *>> $log
    $rc=$LASTEXITCODE
    L "end $w seed$s rc=$rc"
    if($rc -ge 3 -and $rc -le 5){ L "GATE FAILURE rc=$rc - queue stopped"; Set-Content "$base\P15_FAILED" "$w seed$s rc=$rc"; exit $rc }
  }
}
L "=== PHASE 15 queue complete ==="
Set-Content "$base\P15_DONE" (Get-Date -Format s)
