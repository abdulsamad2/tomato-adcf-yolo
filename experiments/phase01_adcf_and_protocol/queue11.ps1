$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue11.log"; $vpy="$repo\.venv\Scripts\python.exe"; $lock="$base\Q11_RUNNING"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
function Done(){ @(Get-ChildItem "$repo\runs_p2b" -Recurse -Filter p2b_done.json -ErrorAction SilentlyContinue).Count }
Set-Location $repo
Set-Content $lock (Get-Date -Format o)
L "=== PHASE 2B pass start (config unchanged; per-run retry added for the WinError 5 dataloader-reset flake) ==="
foreach($seed in 0,1,2,3,4){
  foreach($arm in "B","C","D"){
    Set-Content $lock (Get-Date -Format o)
    if(Test-Path "$repo\runs_p2b\$arm\seed$seed\p2b_done.json"){ L "skip $arm seed$seed"; continue }
    # up to 3 attempts per run. Identical command each time -- the frozen config is never altered,
    # so a retry is a clean rerun of the same arm/seed, not a tuned one.
    for($try=1; $try -le 3; $try++){
      Set-Content $lock (Get-Date -Format o)
      L "--- arm $arm seed $seed attempt $try  (done: $(Done)/15) ---"
      & $vpy "$base\p2b_run.py" --arm $arm --seed $seed *>> $log
      $rc = $LASTEXITCODE
      L "arm $arm seed $seed attempt $try rc=$rc"
      if(Test-Path "$repo\runs_p2b\$arm\seed$seed\p2b_done.json"){ break }
      L "arm $arm seed $seed did not produce p2b_done.json; retrying"
      Start-Sleep -Seconds 20
    }
  }
  L "=== paired seed $seed complete (done: $(Done)/15) ==="
}
$n = Done
Remove-Item $lock -ErrorAction SilentlyContinue
if($n -ge 15){ Set-Content "$base\QUEUE11_DONE" "done"; L "QUEUE11DONE all 15 runs present" }
else { L "INCOMPLETE: $n/15 - exiting nonzero so the watchdog relaunches"; exit 1 }
