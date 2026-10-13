$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"
Set-Location "$base\tomato-adcf-yolo"
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\p14_prov.py" *>> "$base\logs\p14prov.log"
Set-Content "$base\P14PROV_DONE" "done"
