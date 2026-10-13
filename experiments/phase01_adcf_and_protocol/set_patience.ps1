param([int]$P = 25)
$cfg='C:\Users\absamad\cs\tomato-adcf-yolo\configs\experiments.yaml'
Copy-Item $cfg "$cfg.bak" -Force
$c = Get-Content $cfg -Raw
$c = $c -replace '(?m)^(\s*patience:\s*)\d+', ("`${1}" + $P)
[IO.File]::WriteAllText($cfg, $c)
"patience now: " + ((Select-String -Path $cfg -Pattern '^\s*patience:').Line.Trim())
"backup: $cfg.bak"
