$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$log="$base\logs\queue2.log"; $vpy="$repo\.venv\Scripts\python.exe"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
Set-Location $repo
if(-not (Test-Path "$repo\data\tomato_village_noaug\dataset.yaml")){
  L "=== building noaug dataset (same manifest, no _augN copies) ==="
  & $vpy -m adcf_yolo.data.prepare --no-train-aug --out data/tomato_village_noaug --manifest splits/split_manifest.csv *>> $log
  L "prepare rc=$LASTEXITCODE"
}
L "noaug_dataset=" + (Test-Path "$repo\data\tomato_village_noaug\dataset.yaml")
L "=== noaug pair (6 jobs) ==="
& $vpy -m adcf_yolo.experiments --only yolo11s-noaug yolo11s-adcf-noaug --device 0 *>> $log
L "runs rc=$LASTEXITCODE"
& $vpy -m adcf_yolo.collect --out results *>> $log
Set-Content "$base\QUEUE2_DONE" (Get-Date -Format s)
L "=== QUEUE2 COMPLETE ==="
