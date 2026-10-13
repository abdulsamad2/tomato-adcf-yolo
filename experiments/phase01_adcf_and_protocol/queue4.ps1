$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue4.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
L "=== gamma screen: does a larger residual scale let the gate train? (1 seed each) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-adcf-g05 yolo11s-adcf-g10 --seeds 0 --device 0 *>> $log
L "gamma screen rc=$LASTEXITCODE"
foreach($n in @('yolo11s-adcf-g05','yolo11s-adcf-g10')){
  if(Test-Path "$repo\runs\$n\seed0\weights\best.pt"){
    L "--- gate alpha for $n ---"
    & $vpy -m adcf_yolo.gates --weights "runs\$n\seed0\weights\best.pt" --split test --n 2 --out "results\gates_$n" --device cuda:0 *>> $log
  }
}
L "--- GATE C: does either gamma variant beat the 640 baseline by >= 1.5? ---"
& $vpy "$base\gate.py" --a yolo11s --b yolo11s-adcf-g05 yolo11s-adcf-g10 --min-delta 1.5 --winner-file "$base\g_winner.txt" *>> $log
if($LASTEXITCODE -eq 0){
  $w = (Get-Content "$base\g_winner.txt" -Raw).Trim()
  L "GATE C PASS -> confirming $w at 3 seeds"
  & $vpy -m adcf_yolo.experiments --only $w --device 0 *>> $log
  L "confirm rc=$LASTEXITCODE"
} else { L "GATE C FAIL -> gate training was not the blocker; module work stops" }
& $vpy -m adcf_yolo.collect --out results *>> $log
Set-Content "$base\QUEUE4_DONE" (Get-Date -Format s)
L "=== QUEUE4 COMPLETE ==="
