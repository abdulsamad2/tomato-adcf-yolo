$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"; $log="$base\logs\timing.log"
New-Item -ItemType Directory -Force "$base\ultra" | Out-Null
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss),$m) }
$vpy="$repo\.venv\Scripts\python.exe"
Set-Location $repo
L "train yolo11s 3 epochs device 0"
& $vpy -m adcf_yolo.experiments --config configs/timing.yaml --only yolo11s --seeds 0 --device 0 *>> $log
L "train rc=$LASTEXITCODE"
L "ALLDONE"
