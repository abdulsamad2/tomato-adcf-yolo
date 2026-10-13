$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue10.log"; $vpy="$repo\.venv\Scripts\python.exe"; $lock="$base\Q10_RUNNING"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
function Done(){ @(Get-ChildItem "$repo\runs_bcd" -Recurse -Filter bcd_done.json -ErrorAction SilentlyContinue).Count }
Set-Location $repo
Set-Content $lock (Get-Date -Format o)
L "waiting for QUEUE7_DONE (nano queue + idle-card latency benchmarks) before starting"
$w=0
while(-not (Test-Path "$base\QUEUE7_DONE")){
  Set-Content $lock (Get-Date -Format o)      # refresh so the watchdog does not judge us stale
  Start-Sleep -Seconds 60; $w++
  if($w -gt 600){ L "wait timeout after 10 h"; break }
}
L "=== CONTROLLED AUGMENTATION-BUDGET EXPERIMENT: B/C/D x 3 paired seeds ==="
L "FROZEN CONFIG: 49770 updates/arm, no early stopping, primary endpoint = last.pt at update 49770"
L "paired order: seed 0 (B,C,D), then seed 1, then seed 2. No tuning between seeds."
foreach($seed in 0,1,2){
  foreach($arm in "B","C","D"){
    Set-Content $lock (Get-Date -Format o)
    if(Test-Path "$repo\runs_bcd\$arm\seed$seed\bcd_done.json"){ L "skip $arm seed$seed (already complete)"; continue }
    L "--- arm $arm seed $seed  (done so far: $(Done)/9) ---"
    & $vpy "$base\bcd_run.py" --arm $arm --seed $seed *>> $log
    L "arm $arm seed $seed rc=$LASTEXITCODE"
  }
  L "=== paired seed $seed complete (done: $(Done)/9) ==="
}
$n = Done
Remove-Item $lock -ErrorAction SilentlyContinue
if($n -ge 9){ Set-Content "$base\QUEUE10_DONE" "done"; L "QUEUE10DONE all 9 runs present" }
else { L "INCOMPLETE: only $n/9 runs produced bcd_done.json - exiting nonzero so the watchdog retries" ; exit 1 }
