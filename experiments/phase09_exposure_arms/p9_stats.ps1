$base='C:\Users\absamad\cs'; $repo="$base\tomato-adcf-yolo"
$env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
Set-Location $repo
Add-Content "$base\logs\p9stats.log" ("[{0}] start" -f (Get-Date -Format 'MM-dd HH:mm:ss'))
& "$repo\.venv\Scripts\python.exe" "$base\p9_stats.py" *>> "$base\logs\p9stats.log"
Add-Content "$base\logs\p9stats.log" ("[{0}] rc={1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $LASTEXITCODE)
Set-Content "$base\P9STATS_DONE" "done"
