$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue12.log"; $vpy="$repo\.venv\Scripts\python.exe"; $lock="$base\Q12_RUNNING"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
function Done(){ @(Get-ChildItem "$repo\runs_p3" -Recurse -Filter p3_done.json -ErrorAction SilentlyContinue).Count }
Set-Location $repo
Set-Content $lock (Get-Date -Format o)
L "=== PHASE 3 pass start: capacity scaling, yolo11n + yolo11m x 5 paired seeds (s reused from Phase 2B arm D) ==="
L "FROZEN: 39 ep x 630 = 24570 upd; warmup 933; close_mosaic 5; ckpts 6300/11970/18270/24570; primary endpoint 18270"
L "No early stopping. No per-arm tuning. yolo11s NOT retrained. Test untouched."
foreach($seed in 0,1,2,3,4){
  foreach($sz in "n","m"){
    Set-Content $lock (Get-Date -Format o)
    if(Test-Path "$repo\runs_p3\yolo11$sz\seed$seed\p3_done.json"){ L "skip yolo11$sz seed$seed"; continue }
    for($try=1; $try -le 3; $try++){
      Set-Content $lock (Get-Date -Format o)
      L "--- yolo11$sz seed $seed attempt $try  (done: $(Done)/10) ---"
      & $vpy "$base\p3_run.py" --size $sz --seed $seed *>> $log
      L "yolo11$sz seed $seed attempt $try rc=$LASTEXITCODE"
      if(Test-Path "$repo\runs_p3\yolo11$sz\seed$seed\p3_done.json"){ break }
      L "yolo11$sz seed $seed produced no p3_done.json; retrying with the identical command"
      Start-Sleep -Seconds 20
    }
  }
  L "=== paired seed $seed complete (done: $(Done)/10) ==="
}
$n = Done
Remove-Item $lock -ErrorAction SilentlyContinue
if($n -ge 10){ Set-Content "$base\QUEUE12_DONE" "done"; L "QUEUE12DONE all 10 runs present" }
else { L "INCOMPLETE: $n/10 - exiting nonzero so the watchdog relaunches"; exit 1 }
