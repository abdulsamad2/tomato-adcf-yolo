$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"; $log="$base\logs\tier1.log"
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
$env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss),$m) }
$vpy="$repo\.venv\Scripts\python.exe"
Set-Location $repo
New-Item -ItemType Directory -Force "$repo\splits" | Out-Null
Copy-Item "$repo\data\tomato_village\split_manifest.csv" "$repo\splits\" -Force
L "dry-run"
& $vpy -m adcf_yolo.experiments --tier 1 --seeds 0 --device 0 --dry-run *>> $log
L "tier1 seed0 start"
& $vpy -m adcf_yolo.experiments --tier 1 --seeds 0 --device 0 *>> $log
L "tier1 rc=$LASTEXITCODE"
L "ALLDONE"
