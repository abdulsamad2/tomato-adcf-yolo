$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue13.log"; $vpy="$repo\.venv\Scripts\python.exe"; $lock="$base\Q13_RUNNING"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
function Done(){ @(Get-ChildItem "$repo\runs_p4" -Recurse -Filter p4_done.json -ErrorAction SilentlyContinue).Count }
Set-Location $repo
Set-Content $lock (Get-Date -Format o)
L "=== PHASE 4 pass start: YOLO11n resolution scaling 320/416/512 x 5 paired seeds (640 reused from Phase 3) ==="
L "FROZEN: 39 ep x 630 = 24570 upd; warmup 933; close_mosaic 5; ckpts 6300/11970/18270/24570; primary 18270"
L "imgsz is the only variable. No early stopping. 640 NOT retrained. Test untouched."
foreach($seed in 0,1,2,3,4){
  foreach($sz in 320,416,512){
    Set-Content $lock (Get-Date -Format o)
    if(Test-Path "$repo\runs_p4\r$sz\seed$seed\p4_done.json"){ L "skip r$sz seed$seed"; continue }
    for($try=1; $try -le 3; $try++){
      Set-Content $lock (Get-Date -Format o)
      L "--- r$sz seed $seed attempt $try  (done: $(Done)/15) ---"
      & $vpy "$base\p4_run.py" --imgsz $sz --seed $seed *>> $log
      L "r$sz seed $seed attempt $try rc=$LASTEXITCODE"
      if(Test-Path "$repo\runs_p4\r$sz\seed$seed\p4_done.json"){ break }
      L "r$sz seed $seed produced no p4_done.json; retrying with the byte-identical command"
      Start-Sleep -Seconds 20
    }
  }
  L "=== paired seed $seed complete (done: $(Done)/15) ==="
}
$n = Done
if($n -ge 15){
  L "--- all 15 present; computing size-stratified AP for the reused 640 checkpoints (eval only) ---"
  & $vpy "$base\p4_size640.py" *>> $log
  L "size640 rc=$LASTEXITCODE"
  Remove-Item $lock -ErrorAction SilentlyContinue
  Set-Content "$base\QUEUE13_DONE" "done"; L "QUEUE13DONE all 15 runs present"
} else {
  Remove-Item $lock -ErrorAction SilentlyContinue
  L "INCOMPLETE: $n/15 - exiting nonzero so the watchdog relaunches"; exit 1
}
