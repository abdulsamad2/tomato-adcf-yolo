$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"; $log="$base\logs\phaseA.log"
$env:Path="$base\mingit\cmd;"+$env:Path
$env:YOLO_CONFIG_DIR="$base\ultra"; $env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss),$m) }
$vpy="$repo\.venv\Scripts\python.exe"; $sc="$repo\.venv\Scripts"
Set-Location $repo
L "pytest"
& $vpy -m pytest -q *>> $log
L "pytest rc=$LASTEXITCODE"
L "tv-download"
& "$sc\tv-download.exe" *>> $log
L "download rc=$LASTEXITCODE"
L "tv-prepare"
& "$sc\tv-prepare.exe" *>> $log
L "prepare rc=$LASTEXITCODE"
L "tv-prepare-cv"
& "$sc\tv-prepare.exe" --cv-folds 5 *>> $log
L "cv rc=$LASTEXITCODE"
L "tv-stats"
& "$sc\tv-stats.exe" *>> $log
L "stats rc=$LASTEXITCODE"
L "ALLDONE"
