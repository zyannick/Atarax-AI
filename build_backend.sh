#!/bin/bash

set -e

UI_DIR="ataraxai-ui"
TAURI_RESOURCE_DIR="$UI_DIR/src-tauri/py_src"
BUILD_DIR="build"
DIST_DIR="dist"
VENV_DIR=".venv_build"

USE_CUDA=0
CUDA_ARCH=""
SETUP_ARGS=""
CMAKE_ARGS_STR=""


for arg in "$@"; do
    case $arg in
    --use-cuda)
        USE_CUDA=1
        SETUP_ARGS+=" --use-cuda"
        ;;
    --cuda-arch=*)
        CUDA_ARCH="${arg#*=}"
        SETUP_ARGS+=" --cuda-arch=${CUDA_ARCH}"
        ;;
    *)
        echo "Unknown option: $arg. Supported options: --use-cuda, --cuda-arch=<arch>"
        exit 1
        ;;
    esac
done

pkill -f "py_src/api" || echo "No running backend process found to kill."
rm -rf "$BUILD_DIR"
rm -rf "$DIST_DIR"
rm -rf "$TAURI_RESOURCE_DIR"
rm -rf ataraxai.egg-info _skbuild "$VENV_DIR"

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed or not in PATH."
    exit 1
fi
uv venv "$VENV_DIR" --clear
source "$VENV_DIR/bin/activate"
uv pip install --no-cache-dir cmake scikit-build ninja pyinstaller


if [ -f "setup_third_party.sh" ]; then
    ./setup_third_party.sh ${SETUP_ARGS}
else
    echo "Warning: setup_third_party.sh not found."
fi


if [[ $USE_CUDA -eq 1 ]]; then
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=ON"
    if [ -n "$CUDA_ARCH" ]; then
        CMAKE_ARGS_STR+=" -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    fi
else
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=OFF"
fi
CMAKE_ARGS_STR+=" -DGGML_ARM_I8MM=OFF"

PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
ABS_BUILD_DIR=$(pwd)/$BUILD_DIR
cmake -S . -B "$ABS_BUILD_DIR" \
    -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
    ${CMAKE_ARGS_STR}


cmake --build "$ABS_BUILD_DIR" --config Release -- -j $(nproc)


export CMAKE_ARGS="${CMAKE_ARGS_STR}"
uv pip install --no-cache-dir -e .


CPP_EXTENSION_PATH=$(find "$ABS_BUILD_DIR" -name "hegemonikon_py*.so" -print -quit)
if [ -z "$CPP_EXTENSION_PATH" ]; then
    echo "ERROR: Could not find compiled C++ extension (hegemonikon_py.so) in $ABS_BUILD_DIR. Build failed."
    ls -R "$ABS_BUILD_DIR"
    deactivate
    exit 1
fi
echo "Found C++ extension at: $CPP_EXTENSION_PATH"

PYTHON_LIB_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_SO_NAME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_SHARED_LIB="$PYTHON_LIB_PATH/$PYTHON_SO_NAME"

if [ ! -f "$PYTHON_SHARED_LIB" ]; then
    PYTHON_SHARED_LIB=$(find "$VIRTUAL_ENV/lib" -name "libpython*.so.*" -print -quit)
fi

if [ -z "$PYTHON_SHARED_LIB" ] || [ ! -f "$PYTHON_SHARED_LIB" ]; then
    echo "ERROR: Could not locate Python shared library (libpythonX.Y.so). Build failed."
    deactivate
    exit 1
fi

PYTHON_LIB_DIR=$(dirname "$PYTHON_SHARED_LIB")

pyinstaller --noconfirm \
            --onedir \
            --name "api" \
            --distpath "$DIST_DIR" \
            --add-binary "$CPP_EXTENSION_PATH:ataraxai" \
            --add-binary "$PYTHON_SHARED_LIB:_internal" \
            --hidden-import "ataraxai.hegemonikon_py" \
            --hidden-import "chromadb.telemetry.product.posthog" \
            --hidden-import "chromadb.api.rust" \
            --collect-submodules fastapi \
            --collect-submodules uvicorn \
            --collect-submodules ataraxai \
            --exclude-module pytest \
            --exclude-module mypy \
            --exclude-module ruff \
            --exclude-module IPython \
            --upx-dir=/usr/bin \
            api.py

ARTIFACT_DIR="$DIST_DIR/api"
mkdir -p "$TAURI_RESOURCE_DIR"
cp -r "$ARTIFACT_DIR"/* "$TAURI_RESOURCE_DIR"/

DEV_TARGET_DIR="$UI_DIR/src-tauri/target/debug/py_src"
mkdir -p "$DEV_TARGET_DIR"
cp -r "$ARTIFACT_DIR"/* "$DEV_TARGET_DIR"/

echo "Build artifacts copied to:"
echo "  - $TAURI_RESOURCE_DIR (for production builds)"
echo "  - $DEV_TARGET_DIR (for dev mode)"

deactivate

