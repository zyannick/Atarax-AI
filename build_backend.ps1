$ErrorActionPreference = "Stop"

$UI_DIR = "ataraxai-ui"
$TAURI_RESOURCE_DIR = Join-Path $UI_DIR "src-tauri/py_src"
$BUILD_DIR = "build"
$DIST_DIR = "dist"
$VENV_DIR = ".venv_build"

$USE_CUDA = 0
$CUDA_ARCH = ""
$CMAKE_ARGS_STR = ""

foreach ($arg in $args) {
    if ($arg -eq "--use-cuda") {
        $USE_CUDA = 1
    }
    elseif ($arg -like "--cuda-arch=*") {
        $CUDA_ARCH = $arg.Split("=")[1]
    }
    else {
        Write-Error "Unknown option: $arg. Supported options: --use-cuda, --cuda-arch=<arch>"
        exit 1
    }
}

Write-Host "Checking for existing backend process..."
Stop-Process -Name "api" -Force -ErrorAction SilentlyContinue

Write-Host "Cleaning old build directories..."
Remove-Item -Path $BUILD_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path $DIST_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path $TAURI_RESOURCE_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "ataraxai.egg-info", "_skbuild", $VENV_DIR -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "Error: uv is not installed or not in PATH."
    exit 1
}

Write-Host "Creating build virtual environment..."
uv venv $VENV_DIR -p 3.12 --clear

& "$VENV_DIR\Scripts\Activate.ps1"

Write-Host "Installing build tools..."
uv pip install --no-cache-dir cmake scikit-build ninja pyinstaller

Write-Host "[i] Third-party dependencies are now handled by CMake (FetchContent)."

Write-Host "Configuring CMake arguments..."
if ($USE_CUDA -eq 1) {
    $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=ON"
    if ($CUDA_ARCH) {
        $CMAKE_ARGS_STR += " -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    }
} else {
    $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=OFF"
}
$CMAKE_ARGS_STR += " -DGGML_ARM_I8MM=OFF"

$PYTHON_EXECUTABLE = (Get-Command python).Source
$PYTHON_INCLUDE_DIR = python -c "import sysconfig; print(sysconfig.get_path('include'))"
$ABS_BUILD_DIR = (Join-Path (Get-Location).Path $BUILD_DIR)

cmake -S . -B $ABS_BUILD_DIR `
    -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE `
    -DPython_INCLUDE_DIR=$PYTHON_INCLUDE_DIR `
    $CMAKE_ARGS_STR

Write-Host "Building C++ extension..."
cmake --build $ABS_BUILD_DIR --config Release  -j $env:NUMBER_OF_PROCESSORS

$env:CMAKE_ARGS = $CMAKE_ARGS_STR
Write-Host "Installing Python package..."
uv pip install --no-cache-dir -e .

Write-Host "Locating compiled artifacts..."
$CPP_EXTENSION_PATH = (Get-ChildItem -Path $ABS_BUILD_DIR -Filter "hegemonikon_py*.pyd" -Recurse | Select-Object -First 1).FullName

if (-not $CPP_EXTENSION_PATH) {
    Write-Error "ERROR: Could not find compiled C++ extension (hegemonikon_py*.pyd) anywhere inside '$ABS_BUILD_DIR'. Build failed."
    Write-Host "Dumping directory contents:"
    Get-ChildItem -Path $ABS_BUILD_DIR -Recurse | Select-Object FullName
    deactivate
    exit 1
}


Write-Host "Found C++ extension at: $CPP_EXTENSION_PATH"

$PYTHON_DLL_PATH = (Get-ChildItem -Path $env:VIRTUAL_ENV -Filter "python*.dll" -Recurse | Where-Object { $_.Name -match "python3\d{2}\.dll" } | Select-Object -First 1).FullName
    
if (-not $PYTHON_DLL_PATH) {
    $PYTHON_BASE_DIR = [System.IO.Path]::GetDirectoryName((Get-Command python).Source)
    $PYTHON_DLL_PATH = (Get-ChildItem -Path $PYTHON_BASE_DIR -Filter "python*.dll" | Where-Object { $_.Name -match "python3\d{2}\.dll" } | Select-Object -First 1).FullName
}
    
if (-not $PYTHON_DLL_PATH) {
    Write-Error "ERROR: Could not locate Python shared library (pythonXY.dll). Build failed."
    deactivate
    exit 1
}
Write-Host "Found Python DLL at: $PYTHON_DLL_PATH"

Write-Host "Running PyInstaller..."
pyinstaller --noconfirm `
            --onedir `
            --name "api" `
            --distpath $DIST_DIR `
            --add-binary "$CPP_EXTENSION_PATH:ataraxai" `
            --add-binary "$PYTHON_DLL_PATH:_internal" `
            --hidden-import "ataraxai.hegemonikon_py" `
            --hidden-import "chromadb.telemetry.product.posthog" `
            --hidden-import "chromadb.api.rust" `
            --collect-submodules fastapi `
            --collect-submodules uvicorn `
            --collect-submodules ataraxai `
            --exclude-module pytest `
            --exclude-module mypy `
            --exclude-module ruff `
            --exclude-module IPython `
            api.py

# --- 13. Copy Artifacts ---
Write-Host "Copying artifacts to Tauri directory..."
$ARTIFACT_DIR = Join-Path $DIST_DIR "api"
New-Item -ItemType Directory -Path $TAURI_RESOURCE_DIR -Force

Copy-Item -Path (Join-Path $ARTIFACT_DIR "*") -Destination $TAURI_RESOURCE_DIR -Recurse -Force

$DEV_TARGET_DIR = Join-Path $UI_DIR "src-tauri/target/debug/py_src"
New-Item -ItemType Directory -Path $DEV_TARGET_DIR -Force
Copy-Item -Path (Join-Path $ARTIFACT_DIR "*") -Destination $DEV_TARGET_DIR -Recurse -Force

Write-Host "Build artifacts copied to:"
Write-Host "  - $TAURI_RESOURCE_DIR (for production builds)"
Write-Host "  - $DEV_TARGET_DIR (for dev mode)"

deactivate
Write-Host "Build successful."