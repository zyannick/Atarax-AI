#!/bin/bash



set -e
set -o pipefail

UI_DIR="ataraxai-ui"
TAURI_RESOURCE_DIR="$UI_DIR/src-tauri/py_src"
BUILD_DIR="build"
DIST_DIR="dist"
VENV_DIR=".venv_build"
LOG_FILE="build.log"

USE_CUDA=0
CUDA_ARCH=""
CMAKE_ARGS_STR=""
SKIP_CLEANUP=false
VERBOSE=false
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

REQUIRED_PYTHON_VERSION="3.12"
REQUIRED_CMAKE_VERSION="3.20"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

write_log() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local log_message="[$timestamp] [$level] $message"
    
    case "$level" in
        ERROR)
            echo -e "${RED}${log_message}${NC}" >&2
            ;;
        WARN)
            echo -e "${YELLOW}${log_message}${NC}"
            ;;
        SUCCESS)
            echo -e "${GREEN}${log_message}${NC}"
            ;;
        *)
            echo -e "${log_message}"
            ;;
    esac
    
    echo "$log_message" >> "$LOG_FILE"
}

test_command() {
    command -v "$1" &>/dev/null
}

test_version() {
    local cmd="$1"
    local version_arg="${2:---version}"
    
    if output=$($cmd $version_arg 2>&1); then
        write_log "Detected $cmd version: $output" "INFO"
        return 0
    else
        write_log "Could not determine version for $cmd" "WARN"
        return 1
    fi
}

invoke_command_with_check() {
    local description="$1"
    shift
    local cmd="$@"
    
    write_log "Executing: $description" "INFO"
    if [[ "$VERBOSE" == true ]]; then
        write_log "Command: $cmd" "INFO"
    fi
    
    if ! eval "$cmd"; then
        write_log "$description failed with exit code $?" "ERROR"
        exit 1
    fi
    
    write_log "$description completed successfully" "SUCCESS"
}

show_usage() {
    cat << EOF
AtaraxAI Build Script v$SCRIPT_VERSION

Usage: $0 [OPTIONS]

Options:
  --use-cuda              Enable CUDA support for GPU acceleration
  --cuda-arch=<arch>      Specify CUDA architecture (e.g., 75, 86, 89)
  --skip-cleanup          Skip cleaning old build directories
  --verbose               Enable verbose output
  --parallel-jobs=<n>     Number of parallel build jobs (default: $PARALLEL_JOBS)
  --help                  Show this help message

Examples:
  $0
  $0 --use-cuda --cuda-arch=86
  $0 --skip-cleanup --verbose

EOF
    exit 0
}

cleanup_on_error() {
    write_log "════════════════════════════════════════════════════════════" "ERROR"
    write_log "Build failed! Cleaning up..." "ERROR"
    write_log "════════════════════════════════════════════════════════════" "ERROR"
    write_log "Check $LOG_FILE for detailed information" "ERROR"
    
    if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate 2>/dev/null || true
    fi
    
    exit 1
}


for arg in "$@"; do
    case $arg in
        --use-cuda)
            USE_CUDA=1
            ;;
        --cuda-arch=*)
            CUDA_ARCH="${arg#*=}"
            write_log "CUDA architecture set to: $CUDA_ARCH" "INFO"
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --parallel-jobs=*)
            PARALLEL_JOBS="${arg#*=}"
            ;;
        --help)
            show_usage
            ;;
        *)
            write_log "Unknown option: $arg" "ERROR"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

trap cleanup_on_error ERR


: > "$LOG_FILE"

write_log "Build configuration:" "INFO"
write_log "  - CUDA: $(if [[ $USE_CUDA -eq 1 ]]; then echo 'Enabled'; else echo 'Disabled'; fi)" "INFO"
if [[ -n "$CUDA_ARCH" ]]; then
    write_log "  - CUDA Architecture: $CUDA_ARCH" "INFO"
fi
write_log "  - Parallel Jobs: $PARALLEL_JOBS" "INFO"
write_log "  - Skip Cleanup: $SKIP_CLEANUP" "INFO"
write_log "  - Platform: $(uname -s)" "INFO"


write_log "Checking prerequisites..." "INFO"

declare -A required_tools=(
    ["uv"]="Python package manager (https://github.com/astral-sh/uv)"
    ["cmake"]="CMake build system (https://cmake.org/)"
)

for tool in "${!required_tools[@]}"; do
    if ! test_command "$tool"; then
        write_log "Required tool '$tool' not found: ${required_tools[$tool]}" "ERROR"
        exit 1
    fi
    write_log "Found: $tool" "SUCCESS"
done

test_version "cmake" || true

if test_command nproc; then
    PARALLEL_JOBS=$(nproc)
elif test_command sysctl; then
    PARALLEL_JOBS=$(sysctl -n hw.ncpu)
else
    write_log "Could not detect CPU count, using default: $PARALLEL_JOBS" "WARN"
fi


write_log "Checking for existing backend process..." "INFO"
if pkill -f "py_src/api" 2>/dev/null; then
    write_log "Stopped existing backend process" "SUCCESS"
    sleep 1
else
    write_log "No running backend process found" "INFO"
fi


if [[ "$SKIP_CLEANUP" != true ]]; then
    write_log "Cleaning old build directories..." "INFO"
    
    cleanup_paths=(
        "$BUILD_DIR"
        "$DIST_DIR"
        "$TAURI_RESOURCE_DIR"
        "ataraxai.egg-info"
        "_skbuild"
        "$VENV_DIR"
    )
    
    for path in "${cleanup_paths[@]}"; do
        if [[ -e "$path" ]]; then
            write_log "Removing: $path" "INFO"
            rm -rf "$path"
        fi
    done
else
    write_log "Skipping cleanup (--skip-cleanup flag set)" "WARN"
fi


write_log "Creating build virtual environment..." "INFO"
invoke_command_with_check "Virtual environment creation" \
    "uv venv '$VENV_DIR' -p $REQUIRED_PYTHON_VERSION --seed"

ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    write_log "Virtual environment activation script not found at: $ACTIVATE_SCRIPT" "ERROR"
    exit 1
fi

write_log "Activating virtual environment..." "INFO"
source "$ACTIVATE_SCRIPT"

VENV_PYTHON=$(python -c "import sys; print(sys.prefix)")
if [[ "$VENV_PYTHON" != *"$VENV_DIR"* ]]; then
    write_log "Virtual environment activation failed" "ERROR"
    exit 1
fi
write_log "Virtual environment activated: $VENV_PYTHON" "SUCCESS"


write_log "Installing build tools..." "INFO"
invoke_command_with_check "Build tools installation" \
    "uv pip install --no-cache-dir cmake scikit-build ninja pyinstaller"


write_log "Configuring CMake arguments..." "INFO"

if [[ $USE_CUDA -eq 1 ]]; then
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=ON"
    if [[ -n "$CUDA_ARCH" ]]; then
        CMAKE_ARGS_STR+=" -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    fi
    write_log "CUDA support enabled" "INFO"
else
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=OFF"
fi

CMAKE_ARGS_STR+=" -DGGML_ARM_I8MM=OFF"
CMAKE_ARGS_STR+=" -DCMAKE_POSITION_INDEPENDENT_CODE=ON"

write_log "CMake arguments: $CMAKE_ARGS_STR" "INFO"


PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
ABS_BUILD_DIR=$(pwd)/$BUILD_DIR

write_log "Python executable: $PYTHON_EXECUTABLE" "INFO"
write_log "Python include dir: $PYTHON_INCLUDE_DIR" "INFO"
write_log "Build directory: $ABS_BUILD_DIR" "INFO"

write_log "Configuring CMake project..." "INFO"
invoke_command_with_check "CMake configuration" \
    "cmake -S . -B '$ABS_BUILD_DIR' -DPYTHON_EXECUTABLE='$PYTHON_EXECUTABLE' -DPython_INCLUDE_DIR='$PYTHON_INCLUDE_DIR' $CMAKE_ARGS_STR"

write_log "Building C++ extension with $PARALLEL_JOBS parallel jobs..." "INFO"
invoke_command_with_check "C++ extension build" \
    "cmake --build '$ABS_BUILD_DIR' --config Release -- -j $PARALLEL_JOBS"



export CMAKE_ARGS="${CMAKE_ARGS_STR}"
write_log "Installing Python package..." "INFO"
invoke_command_with_check "Python package installation" \
    "uv pip install --no-cache-dir -e ."


write_log "Locating compiled artifacts..." "INFO"

if [[ ! -d "$ABS_BUILD_DIR" ]]; then
    write_log "Build directory does not exist: $ABS_BUILD_DIR" "ERROR"
    exit 1
fi

write_log "Searching for C++ extension in: $ABS_BUILD_DIR" "INFO"
CPP_EXTENSION_PATH=$(find "$ABS_BUILD_DIR" -name "hegemonikon_py*.so" -print -quit)

if [[ -z "$CPP_EXTENSION_PATH" ]] || [[ ! -f "$CPP_EXTENSION_PATH" ]]; then
    write_log "Could not find compiled C++ extension (hegemonikon_py*.so)" "ERROR"
    write_log "Directory contents of $ABS_BUILD_DIR:" "ERROR"
    ls -R "$ABS_BUILD_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

write_log "Found C++ extension: $CPP_EXTENSION_PATH" "SUCCESS"


write_log "Locating Python shared library..." "INFO"

PYTHON_LIB_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "")
PYTHON_SO_NAME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))" 2>/dev/null || echo "")

if [[ -n "$PYTHON_LIB_PATH" ]] && [[ -n "$PYTHON_SO_NAME" ]]; then
    PYTHON_SHARED_LIB="$PYTHON_LIB_PATH/$PYTHON_SO_NAME"
    write_log "Method 1: Checking $PYTHON_SHARED_LIB" "INFO"
fi

if [[ -z "$PYTHON_SHARED_LIB" ]] || [[ ! -f "$PYTHON_SHARED_LIB" ]]; then
    write_log "Method 1 failed, searching in virtual environment..." "INFO"
    PYTHON_SHARED_LIB=$(find "$VIRTUAL_ENV/lib" -name "libpython*.so.*" -print -quit 2>/dev/null || echo "")
fi

if [[ -z "$PYTHON_SHARED_LIB" ]] || [[ ! -f "$PYTHON_SHARED_LIB" ]]; then
    write_log "Method 2 failed, searching common locations..." "INFO"
    for path in /usr/lib /usr/local/lib /opt/homebrew/lib "$HOME/.pyenv/versions/"*/lib; do
        if [[ -d "$path" ]]; then
            PYTHON_SHARED_LIB=$(find "$path" -name "libpython*.so.*" -o -name "libpython*.dylib" 2>/dev/null | head -n1)
            if [[ -n "$PYTHON_SHARED_LIB" ]] && [[ -f "$PYTHON_SHARED_LIB" ]]; then
                break
            fi
        fi
    done
fi

if [[ -z "$PYTHON_SHARED_LIB" ]] || [[ ! -f "$PYTHON_SHARED_LIB" ]]; then
    write_log "Method 3 failed, using Python to construct path..." "INFO"
    PYTHON_VERSION_SHORT=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    BASE_PREFIX=$(python -c "import sys; print(sys.base_prefix)")
    
    for ext in "so.$PYTHON_VERSION_SHORT" "so" "dylib"; do
        TEST_PATH="$BASE_PREFIX/lib/libpython$PYTHON_VERSION_SHORT.$ext"
        if [[ -f "$TEST_PATH" ]]; then
            PYTHON_SHARED_LIB="$TEST_PATH"
            break
        fi
    done
fi

if [[ -z "$PYTHON_SHARED_LIB" ]] || [[ ! -f "$PYTHON_SHARED_LIB" ]]; then
    write_log "Could not locate Python shared library (libpythonX.Y.so)" "ERROR"
    write_log "Attempted paths:" "ERROR"
    write_log "  - $PYTHON_LIB_PATH/$PYTHON_SO_NAME" "ERROR"
    write_log "  - $VIRTUAL_ENV/lib/libpython*.so.*" "ERROR"
    exit 1
fi

write_log "Found Python shared library: $PYTHON_SHARED_LIB" "SUCCESS"


write_log "Running PyInstaller..." "INFO"

PYINSTALLER_CMD="pyinstaller --noconfirm \
    --onedir \
    --name 'api' \
    --distpath '$DIST_DIR' \
    --add-binary '$CPP_EXTENSION_PATH:ataraxai' \
    --add-binary '$PYTHON_SHARED_LIB:_internal' \
    --hidden-import 'ataraxai.hegemonikon_py' \
    --hidden-import 'chromadb.telemetry.product.posthog' \
    --hidden-import 'chromadb.api.rust' \
    --collect-submodules fastapi \
    --collect-submodules uvicorn \
    --collect-submodules ataraxai \
    --exclude-module pytest \
    --exclude-module mypy \
    --exclude-module ruff \
    --exclude-module IPython"

if test_command upx; then
    PYINSTALLER_CMD+=" --upx-dir=$(dirname $(which upx))"
fi

PYINSTALLER_CMD+=" api.py"

invoke_command_with_check "PyInstaller bundling" "$PYINSTALLER_CMD"

write_log "Copying artifacts to Tauri directories..." "INFO"

ARTIFACT_DIR="$DIST_DIR/api"

if [[ ! -d "$ARTIFACT_DIR" ]]; then
    write_log "Artifact directory not found: $ARTIFACT_DIR" "ERROR"
    exit 1
fi

mkdir -p "$TAURI_RESOURCE_DIR"
cp -r "$ARTIFACT_DIR"/* "$TAURI_RESOURCE_DIR"/
write_log "Artifacts copied to: $TAURI_RESOURCE_DIR" "SUCCESS"

DEV_TARGET_DIR="$UI_DIR/src-tauri/target/debug/py_src"
mkdir -p "$DEV_TARGET_DIR"
cp -r "$ARTIFACT_DIR"/* "$DEV_TARGET_DIR"/
write_log "Artifacts copied to: $DEV_TARGET_DIR" "SUCCESS"


write_log "════════════════════════════════════════════════════════════" "SUCCESS"
write_log "Build completed successfully!" "SUCCESS"
write_log "════════════════════════════════════════════════════════════" "SUCCESS"
write_log "" "INFO"
write_log "Artifacts locations:" "INFO"
write_log "  Production: $TAURI_RESOURCE_DIR" "INFO"
write_log "  Development: $DEV_TARGET_DIR" "INFO"
write_log "" "INFO"
write_log "C++ Extension: $CPP_EXTENSION_PATH" "INFO"
write_log "Python Shared Library: $PYTHON_SHARED_LIB" "INFO"
write_log "" "INFO"
write_log "Build log saved to: $LOG_FILE" "INFO"

deactivate
exit 0