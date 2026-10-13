$base='C:\Users\absamad\cs'; $log="$base\logs\densenet.log"
$env:TORCH_HOME="$base\torchhome"; $env:MPLCONFIGDIR="$base\mpl"; $env:PYTHONUTF8="1"
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
L "=== DenseNet121 on Variant-a: 3 seeds x (official split, leakage-free split) ==="
& "$base\tomato-adcf-yolo\.venv\Scripts\python.exe" "$base\va_densenet.py" *>> $log
L "densenet rc=$LASTEXITCODE"
Set-Content "$base\DENSENET_DONE" (Get-Date -Format s)
