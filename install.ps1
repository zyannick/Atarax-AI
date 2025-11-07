param (
    [switch]$clean,
    [switch]$usecuda,
    [switch]$cleancache,
    [string]$cudaarch = "",
    [switch]$onlycpp,
    [switch]$useconda,
    [switch]$useuv
)

$ErrorActionPreference = "Stop"

$CMAKE_ARGS_STR = ""

if ($cleancache) {
    Write-Host "Cleaning ccache..."
    ccache -C -ErrorAction SilentlyContinue
}

if ($clean) {
    Write-Host "Cleaning old build directories..."
    Remove-Item -Path "build", "ataraxai.egg-info", "dist", "_skbuild" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "ataraxai\*.pyd" -Force -ErrorAction SilentlyContinue
    
    if (Test-Path -Path "./clean.ps1") {
        ./clean.ps1
    }
}

if ($useuv) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "Error: uv is not installed or not in PATH."
        exit 1
    }
    Write-Host "Creating 'uv' virtual environment..."
    uv venv .venv -p 3.12 --clear
    
    .\.venv\Scripts\Activate.ps1
    
    Write-Host "Installing build tools..."
    uv pip install cmake scikit-build ninja
}

if ($useconda) {
    if (-not $env:CONDA_PREFIX) {
        Write-Error "Error: --use-conda specified, but CONDA_PREFIX is not set. Activate Conda first."
        exit 1
    }
    Write-Host "Configuring for Conda environment at $env:CONDA_PREFIX"
}

try {
    if ($useuv) {
        uv pip uninstall ataraxai
    } else {
        pip uninstall ataraxai -y
    }
} catch {
    Write-Warning "Could not uninstall previous version (this is usually fine)."
}

if ($usecuda) {
    $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=ON"
    if ($cudaarch) {
        $CMAKE_ARGS_STR += " -DCMAKE_CUDA_ARCHITECTURES=$cudaarch"
    }
} else {
    $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=OFF"
}

$CMAKE_ARGS_STR += " -DBUILD_TESTING=ON"

$env:CMAKE_ARGS = $CMAKE_ARGS_STR

cmake -S . -B build $CMAKE_ARGS_STR

cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS
cmake --build build --target hegemonikon_tests --config Release

if ($onlycpp) {
    Write-Host "C++ build complete. Exiting due to --only-cpp flag."
    if ($useuv) { 
        deactivate 
    }
    exit 0
}

Write-Host "Installing Python package..."
if ($useuv) {
    uv pip install -e .
} else {
    python -m pip install -e .
}

Write-Host "Verifying installation..."
python -c "from ataraxai import hegemonikon_py; print('[SUCCESS] Atarax-AI installed and core module is importable!')"

if ($useuv) {
    deactivate
    Write-Host "Build complete. Virtual environment deactivated."
}