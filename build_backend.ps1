#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================
$SCRIPT_VERSION = "2.0.0"
$UI_DIR = "ataraxai-ui"
$TAURI_RESOURCE_DIR = Join-Path $UI_DIR "src-tauri/py_src"
$BUILD_DIR = "build"
$DIST_DIR = "dist"
$VENV_DIR = ".venv_build"
$LOG_FILE = "build.log"

# Build options
$USE_CUDA = 0
$CUDA_ARCH = ""
$CMAKE_ARGS_STR = ""
$SKIP_CLEANUP = $false
$VERBOSE = $false
$PARALLEL_JOBS = $env:NUMBER_OF_PROCESSORS

# Required tool versions
$REQUIRED_PYTHON_VERSION = "3.12"
$REQUIRED_CMAKE_VERSION = "3.20"

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "WARN"  { Write-Host $logMessage -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $logMessage -ForegroundColor Green }
        default { Write-Host $logMessage }
    }
    
    Add-Content -Path $LOG_FILE -Value $logMessage
}

function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-Version {
    param(
        [string]$Command,
        [string]$RequiredVersion,
        [string]$VersionArg = "--version"
    )
    try {
        $output = & $Command $VersionArg 2>&1 | Out-String
        Write-Log "Detected $Command version: $output" "INFO"
        return $true
    }
    catch {
        Write-Log "Could not determine version for $Command" "WARN"
        return $false
    }
}

function Invoke-CommandWithCheck {
    param(
        [string]$Command,
        [string]$Description
    )
    Write-Log "Executing: $Description" "INFO"
    if ($VERBOSE) {
        Write-Log "Command: $Command" "INFO"
    }
    
    Invoke-Expression $Command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "$Description failed with exit code $LASTEXITCODE" "ERROR"
        throw "$Description failed"
    }
    Write-Log "$Description completed successfully" "SUCCESS"
}

function Show-Usage {
    Write-Host @"
AtaraxAI Build Script v$SCRIPT_VERSION

Usage: .\build.ps1 [OPTIONS]

Options:
  --use-cuda              Enable CUDA support for GPU acceleration
  --cuda-arch=<arch>      Specify CUDA architecture (e.g., 75, 86, 89)
  --skip-cleanup          Skip cleaning old build directories
  --verbose               Enable verbose output
  --parallel-jobs=<n>     Number of parallel build jobs (default: $env:NUMBER_OF_PROCESSORS)
  --help                  Show this help message

Examples:
  .\build.ps1
  .\build.ps1 --use-cuda --cuda-arch=86
  .\build.ps1 --skip-cleanup --verbose

"@
    exit 0
}

# ============================================================================
# Parse Arguments
# ============================================================================

foreach ($arg in $args) {
    switch -Regex ($arg) {
        "^--use-cuda$" { 
            $USE_CUDA = 1 
        }
        "^--cuda-arch=(.+)$" { 
            $CUDA_ARCH = $Matches[1]
            Write-Log "CUDA architecture set to: $CUDA_ARCH" "INFO"
        }
        "^--skip-cleanup$" { 
            $SKIP_CLEANUP = $true 
        }
        "^--verbose$" { 
            $VERBOSE = $true 
        }
        "^--parallel-jobs=(\d+)$" { 
            $PARALLEL_JOBS = $Matches[1]
        }
        "^--help$" { 
            Show-Usage 
        }
        default {
            Write-Log "Unknown option: $arg" "ERROR"
            Write-Host "Use --help for usage information"
            exit 1
        }
    }
}

# ============================================================================
# Main Build Process
# ============================================================================

try {
    Write-Log "Starting AtaraxAI build process (v$SCRIPT_VERSION)" "INFO"
    Write-Log "Build configuration:" "INFO"
    Write-Log "  - CUDA: $(if ($USE_CUDA) { 'Enabled' } else { 'Disabled' })" "INFO"
    if ($CUDA_ARCH) {
        Write-Log "  - CUDA Architecture: $CUDA_ARCH" "INFO"
    }
    Write-Log "  - Parallel Jobs: $PARALLEL_JOBS" "INFO"
    Write-Log "  - Skip Cleanup: $SKIP_CLEANUP" "INFO"

    # ========================================================================
    # Check Prerequisites
    # ========================================================================
    
    Write-Log "Checking prerequisites..." "INFO"
    
    $requiredTools = @{
        "uv" = "Python package manager (https://github.com/astral-sh/uv)"
        "cmake" = "CMake build system (https://cmake.org/)"
        "python" = "Python interpreter"
    }
    
    foreach ($tool in $requiredTools.Keys) {
        if (-not (Test-Command $tool)) {
            Write-Log "Required tool '$tool' not found: $($requiredTools[$tool])" "ERROR"
            exit 1
        }
        Write-Log "Found: $tool" "SUCCESS"
    }
    
    # Check Python version
    $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Log "Python version: $pythonVersion" "INFO"
    
    # Check CMake version
    Test-Version "cmake" $REQUIRED_CMAKE_VERSION | Out-Null

    # ========================================================================
    # Stop Existing Processes
    # ========================================================================
    
    Write-Log "Checking for existing backend process..." "INFO"
    $existingProcess = Get-Process -Name "api" -ErrorAction SilentlyContinue
    if ($existingProcess) {
        Write-Log "Stopping existing 'api' process (PID: $($existingProcess.Id))..." "WARN"
        Stop-Process -Name "api" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    # ========================================================================
    # Cleanup
    # ========================================================================
    
    if (-not $SKIP_CLEANUP) {
        Write-Log "Cleaning old build directories..." "INFO"
        
        $cleanupPaths = @(
            $BUILD_DIR,
            $DIST_DIR,
            $TAURI_RESOURCE_DIR,
            "ataraxai.egg-info",
            "_skbuild",
            $VENV_DIR
        )
        
        foreach ($path in $cleanupPaths) {
            if (Test-Path $path) {
                Write-Log "Removing: $path" "INFO"
                Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    } else {
        Write-Log "Skipping cleanup (--skip-cleanup flag set)" "WARN"
    }

    # ========================================================================
    # Create Virtual Environment
    # ========================================================================
    
    Write-Log "Creating build virtual environment..." "INFO"
    Invoke-CommandWithCheck "uv venv $VENV_DIR -p $REQUIRED_PYTHON_VERSION --seed" "Virtual environment creation"
    
    # Activate virtual environment
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        throw "Virtual environment activation script not found at: $activateScript"
    }
    
    Write-Log "Activating virtual environment..." "INFO"
    & $activateScript
    
    # Verify activation
    $venvPython = & python -c "import sys; print(sys.prefix)"
    if ($venvPython -notlike "*$VENV_DIR*") {
        throw "Virtual environment activation failed"
    }
    Write-Log "Virtual environment activated: $venvPython" "SUCCESS"

    # ========================================================================
    # Install Build Tools
    # ========================================================================
    
    Write-Log "Installing build tools..." "INFO"
    Invoke-CommandWithCheck "uv pip install --no-cache-dir cmake scikit-build ninja pyinstaller" "Build tools installation"

    # ========================================================================
    # Configure CMake
    # ========================================================================
    
    Write-Log "Configuring CMake arguments..." "INFO"
    
    if ($USE_CUDA -eq 1) {
        $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=ON"
        if ($CUDA_ARCH) {
            $CMAKE_ARGS_STR += " -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
        }
        Write-Log "CUDA support enabled" "INFO"
    } else {
        $CMAKE_ARGS_STR = "-DATARAXAI_USE_CUDA=OFF"
    }
    $CMAKE_ARGS_STR += " -DGGML_ARM_I8MM=OFF"
    
    Write-Log "CMake arguments: $CMAKE_ARGS_STR" "INFO"

    # ========================================================================
    # Build C++ Extension
    # ========================================================================
    
    $PYTHON_EXECUTABLE = (Get-Command python).Source
    $PYTHON_INCLUDE_DIR = python -c "import sysconfig; print(sysconfig.get_path('include'))"
    $ABS_BUILD_DIR = Join-Path (Get-Location).Path $BUILD_DIR
    
    Write-Log "Python executable: $PYTHON_EXECUTABLE" "INFO"
    Write-Log "Python include dir: $PYTHON_INCLUDE_DIR" "INFO"
    Write-Log "Build directory: $ABS_BUILD_DIR" "INFO"
    
    Write-Log "Configuring CMake project..." "INFO"
    $cmakeConfigCmd = "cmake -S . -B `"$ABS_BUILD_DIR`" -DPYTHON_EXECUTABLE=`"$PYTHON_EXECUTABLE`" -DPython_INCLUDE_DIR=`"$PYTHON_INCLUDE_DIR`" $CMAKE_ARGS_STR"
    Invoke-CommandWithCheck $cmakeConfigCmd "CMake configuration"
    
    Write-Log "Building C++ extension with $PARALLEL_JOBS parallel jobs..." "INFO"
    $cmakeBuildCmd = "cmake --build `"$ABS_BUILD_DIR`" --config Release -j $PARALLEL_JOBS"
    Invoke-CommandWithCheck $cmakeBuildCmd "C++ extension build"

    # ========================================================================
    # Install Python Package
    # ========================================================================
    
    $env:CMAKE_ARGS = $CMAKE_ARGS_STR
    Write-Log "Installing Python package..." "INFO"
    Invoke-CommandWithCheck "uv pip install --no-cache-dir -e ." "Python package installation"

    # ========================================================================
    # Locate Compiled Artifacts
    # ========================================================================
    
    Write-Log "Locating compiled artifacts..." "INFO"
    
    $SEARCH_PATH = Join-Path $VENV_DIR "Lib\site-packages\ataraxai"
    
    if (-not (Test-Path $SEARCH_PATH)) {
        throw "Search path does not exist: $SEARCH_PATH"
    }
    
    Write-Log "Searching for C++ extension in: $SEARCH_PATH" "INFO"
    $CPP_EXTENSION_PATH = (Get-ChildItem -Path $SEARCH_PATH -Filter "hegemonikon_py*.pyd" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1)
    
    if (-not $CPP_EXTENSION_PATH) {
        Write-Log "Could not find compiled C++ extension (hegemonikon_py*.pyd)" "ERROR"
        Write-Log "Directory contents of $SEARCH_PATH :" "ERROR"
        Get-ChildItem -Path $SEARCH_PATH -Recurse | Select-Object FullName | Format-Table -AutoSize | Out-String | Write-Log
        throw "C++ extension not found"
    }
    
    $CPP_EXTENSION_PATH = $CPP_EXTENSION_PATH.FullName
    Write-Log "Found C++ extension: $CPP_EXTENSION_PATH" "SUCCESS"

    # ========================================================================
    # Locate Python DLL
    # ========================================================================
    
    Write-Log "Locating Python shared library..." "INFO"
    
    $PYTHON_DLL_PATH = (Get-ChildItem -Path $env:VIRTUAL_ENV -Filter "python*.dll" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "python3\d{2}\.dll" } | Select-Object -First 1)
    
    if (-not $PYTHON_DLL_PATH) {
        $PYTHON_BASE_DIR = [System.IO.Path]::GetDirectoryName((Get-Command python).Source)
        Write-Log "Searching Python base directory: $PYTHON_BASE_DIR" "INFO"
        $PYTHON_DLL_PATH = (Get-ChildItem -Path $PYTHON_BASE_DIR -Filter "python*.dll" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "python3\d{2}\.dll" } | Select-Object -First 1)
    }
    
    if (-not $PYTHON_DLL_PATH) {
        Write-Log "Trying to locate DLL using Python..." "INFO"
        $dllName = python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}.dll')"
        $dllPath = python -c "import sys, os; print(os.path.join(sys.base_prefix, '$dllName'))"
        if (Test-Path $dllPath) {
            $PYTHON_DLL_PATH = Get-Item $dllPath
        }
    }
    
    if (-not $PYTHON_DLL_PATH) {
        throw "Could not locate Python shared library (pythonXY.dll)"
    }
    
    $PYTHON_DLL_PATH = $PYTHON_DLL_PATH.FullName
    Write-Log "Found Python DLL: $PYTHON_DLL_PATH" "SUCCESS"

    # ========================================================================
    # Run PyInstaller
    # ========================================================================
    
    Write-Log "Running PyInstaller..." "INFO"
    
    $pyinstallerCmd = @"
pyinstaller --noconfirm ``
            --onedir ``
            --name "api" ``
            --distpath "$DIST_DIR" ``
            --add-binary "$CPP_EXTENSION_PATH;ataraxai" ``
            --add-binary "$PYTHON_DLL_PATH;_internal" ``
            --hidden-import "ataraxai.hegemonikon_py" ``
            --hidden-import "chromadb.telemetry.product.posthog" ``
            --hidden-import "chromadb.api.rust" ``
            --collect-submodules fastapi ``
            --collect-submodules uvicorn ``
            --collect-submodules ataraxai ``
            --exclude-module pytest ``
            --exclude-module mypy ``
            --exclude-module ruff ``
            --exclude-module IPython ``
            api.py
"@
    
    Invoke-CommandWithCheck $pyinstallerCmd "PyInstaller bundling"

    # ========================================================================
    # Copy Artifacts to Tauri
    # ========================================================================
    
    Write-Log "Copying artifacts to Tauri directories..." "INFO"
    
    $ARTIFACT_DIR = Join-Path $DIST_DIR "api"
    
    if (-not (Test-Path $ARTIFACT_DIR)) {
        throw "Artifact directory not found: $ARTIFACT_DIR"
    }
    
    # Production build directory
    New-Item -ItemType Directory -Path $TAURI_RESOURCE_DIR -Force | Out-Null
    Copy-Item -Path (Join-Path $ARTIFACT_DIR "*") -Destination $TAURI_RESOURCE_DIR -Recurse -Force
    Write-Log "Artifacts copied to: $TAURI_RESOURCE_DIR" "SUCCESS"
    
    # Development build directory
    $DEV_TARGET_DIR = Join-Path $UI_DIR "src-tauri/target/debug/py_src"
    New-Item -ItemType Directory -Path $DEV_TARGET_DIR -Force | Out-Null
    Copy-Item -Path (Join-Path $ARTIFACT_DIR "*") -Destination $DEV_TARGET_DIR -Recurse -Force
    Write-Log "Artifacts copied to: $DEV_TARGET_DIR" "SUCCESS"

    # ========================================================================
    # Build Summary
    # ========================================================================
    
    Write-Log "Build completed successfully!" "SUCCESS"
    Write-Log "" "INFO"
    Write-Log "Artifacts locations:" "INFO"
    Write-Log "  Production: $TAURI_RESOURCE_DIR" "INFO"
    Write-Log "  Development: $DEV_TARGET_DIR" "INFO"
    Write-Log "" "INFO"
    Write-Log "C++ Extension: $CPP_EXTENSION_PATH" "INFO"
    Write-Log "Python DLL: $PYTHON_DLL_PATH" "INFO"
    Write-Log "" "INFO"
    Write-Log "Build log saved to: $LOG_FILE" "INFO"
    
    deactivate
    exit 0

} catch {
    Write-Log "═══════════════════════════════════════════════════════════" "ERROR"
    Write-Log "Build failed: $($_.Exception.Message)" "ERROR"
    Write-Log "═══════════════════════════════════════════════════════════" "ERROR"
    Write-Log "Stack trace:" "ERROR"
    Write-Log $_.ScriptStackTrace "ERROR"
    Write-Log "" "ERROR"
    Write-Log "Check $LOG_FILE for detailed information" "ERROR"
    
    # Attempt to deactivate venv if active
    try { deactivate } catch {}
    
    exit 1
}