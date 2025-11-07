Write-Host "Cleaning previous build artifacts..."

Remove-Item -Path "build", "_skbuild", "ataraxai.egg-info", "dist", ".venv" -Recurse -Force -ErrorAction SilentlyContinue

Remove-Item -Path "ataraxai\hegemonikon_py*.pyd" -Force -ErrorAction SilentlyContinue

Write-Host "Cleaning complete."