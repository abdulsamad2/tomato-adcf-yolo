$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue14.log"; $vpy="$repo\.venv\Scripts\python.exe"; $lock="$base\Q14_RUNNING"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
function Done(){ @(Get-ChildItem "$repo\runs_p6" -Recurse -Filter p6_done.json -ErrorAction SilentlyContinue).Count }
Set-Location $repo
Set-Content $lock (Get-Date -Format o)
L "=== PHASE 6 start: detection-scale B (P3+P4) and C (P4+P5), seeds 0-4. A reused from Phase 4. ==="
L "FROZEN: 512px, 39 ep x 630 = 24570 upd, warmup 933, close_mosaic 5, ckpts 6300/11970/18270/24570, primary 18270"
L "Init: audited remap, verified by tensor-value equality in on_train_start; mismatch aborts the run."
foreach($seed in 0,1,2,3,4){
  foreach($arm in "B","C"){
    Set-Content $lock (Get-Date -Format o)
    if(Test-Path "$repo\runs_p6\$arm\seed$seed\p6_done.json"){ L "skip $arm seed$seed"; continue }
    for($try=1; $try -le 3; $try++){
      Set-Content $lock (Get-Date -Format o)
      $g = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
      L "--- $arm seed $seed attempt $try  (done: $(Done)/10) gpu=$g ---"
      & $vpy "$base\p6_run.py" --arm $arm --seed $seed *>> $log
      $rc = $LASTEXITCODE
      L "$arm seed $seed attempt $try rc=$rc"
      if($rc -eq 2){ L "$arm seed $seed INIT GATE FAILED - not retrying, this is not an infrastructure failure"; break }
      if(Test-Path "$repo\runs_p6\$arm\seed$seed\p6_done.json"){ break }
      L "$arm seed $seed produced no p6_done.json; retrying with the byte-identical command"
      Start-Sleep -Seconds 20
    }
  }
  L "=== paired seed $seed complete (done: $(Done)/10) ==="
}
$n = Done
Remove-Item $lock -ErrorAction SilentlyContinue
if($n -ge 10){ Set-Content "$base\QUEUE14_DONE" "done"; L "QUEUE14DONE all 10 runs present" }
else { L "INCOMPLETE: $n/10 - exiting nonzero so the watchdog relaunches"; exit 1 }
