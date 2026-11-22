python "$PSScriptRoot\phase3_mfe_fnn.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot\phase4_cnn.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot\phase5_stacking.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot\phase6_compare.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
