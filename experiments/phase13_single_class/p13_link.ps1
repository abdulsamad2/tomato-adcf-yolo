$base='C:\Users\absamad\cs'
$env:PYTHONUTF8="1"
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\p13_link.py" *>> "$base\logs\p13link.log"
Set-Content "$base\P13LINK_DONE" "done"
