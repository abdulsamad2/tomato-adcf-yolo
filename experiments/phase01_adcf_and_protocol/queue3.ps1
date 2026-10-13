$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue3.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo

if(-not (Test-Path "$repo\results\gates\gate_stats.json")){
  L "=== stage 0: gate interpretability (alpha inside vs outside boxes, no training) ==="
  & $vpy -m adcf_yolo.gates --weights "runs\yolo11s-adcf\seed0\weights\best.pt" --split test --n 6 --out "results\gates" --device cuda:0 *>> $log
  L "gates rc=$LASTEXITCODE"
}

L "=== stage 1: 960 px screen, 1 seed each (~6 h) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-960 yolo11s-adcf-960 --seeds 0 --device 0 *>> $log
L "screen960 rc=$LASTEXITCODE"
L "--- GATE A: ADCF vs matched 960 baseline, threshold +1.5 ---"
& $vpy "$base\gate.py" --a yolo11s-960 --b yolo11s-adcf-960 --min-delta 1.5 *>> $log
if($LASTEXITCODE -eq 0){
  L "GATE A PASS -> confirming both at 3 seeds"
  & $vpy -m adcf_yolo.experiments --only yolo11s-960 yolo11s-adcf-960 --device 0 *>> $log
  L "confirm960 rc=$LASTEXITCODE"
} else { L "GATE A FAIL -> no 960 confirmation; module work stops here" }

L "=== stage 2: copy-paste screen, 1 seed each (~3 h) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-cp10 yolo11s-cp30 --seeds 0 --device 0 *>> $log
L "cpscreen rc=$LASTEXITCODE"
L "--- GATE B: copy-paste vs 640 baseline, threshold +1.5 ---"
& $vpy "$base\gate.py" --a yolo11s --b yolo11s-cp10 yolo11s-cp30 --min-delta 1.5 --winner-file "$base\cp_winner.txt" *>> $log
if($LASTEXITCODE -eq 0){
  $w = (Get-Content "$base\cp_winner.txt" -Raw).Trim()
  L "GATE B PASS -> confirming $w at 3 seeds"
  & $vpy -m adcf_yolo.experiments --only $w --device 0 *>> $log
  L "confirmcp rc=$LASTEXITCODE"
} else { L "GATE B FAIL -> no copy-paste confirmation" }

L "=== stage 3: collect ==="
& $vpy -m adcf_yolo.collect --out results *>> $log
Set-Content "$base\QUEUE3_DONE" (Get-Date -Format s)
L "=== QUEUE3 COMPLETE ==="
