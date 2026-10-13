$base='C:\Users\absamad\cs'; $aud="$base\audit"; $log="$base\logs\fetch.log"
$env:Path="$base\mingit\cmd;"+$env:Path
function L($m){ Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'),$m) }
New-Item -ItemType Directory -Force $aud | Out-Null
L "start"

# 1. Tomato-Village full repo (Variant-a and Variant-b, same authors as Variant-c)
if(-not (Test-Path "$aud\Tomato-Village\.git")){
  L "cloning Tomato-Village (full)"
  & git clone --filter=blob:none --no-checkout https://github.com/mamta-joshi-gehlot/Tomato-Village.git "$aud\Tomato-Village" *>> $log
  Set-Location "$aud\Tomato-Village"
  & git sparse-checkout init --cone *>> $log
  & git sparse-checkout set "Variant-a(Classification)" "Variant-b(Classification)" *>> $log
  & git checkout *>> $log
  L "TV full rc=$LASTEXITCODE"
} else { L "TV already present" }

# 2. PlantDoc detection dataset
if(-not (Test-Path "$aud\PlantDoc-Dataset\.git")){
  L "cloning PlantDoc"
  & git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git "$aud\PlantDoc-Dataset" *>> $log
  L "PlantDoc rc=$LASTEXITCODE"
} else { L "PlantDoc already present" }

L "--- inventory ---"
foreach($d in (Get-ChildItem $aud -Directory)){
  $n = @(Get-ChildItem $d.FullName -Recurse -File -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
  L ("{0} = {1} images" -f $d.Name, $n)
}
L "FETCHDONE"
