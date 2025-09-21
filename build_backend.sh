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

echo "--- Starting Atarax-AI Backend Build for Tauri ---"

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


rm -rf "$BUILD_DIR"
rm -rf "$DIST_DIR"
rm -rf "$TAURI_RESOURCE_DIR"
rm -rf ataraxai.egg-info _skbuild "$VENV_DIR"
find . -name "*.so" -delete

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

PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")

cmake -S . -B "$BUILD_DIR" \
    -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
    ${CMAKE_ARGS_STR}

cmake --build "$BUILD_DIR" --config Release -- -j$(nproc)

uv pip install --no-cache-dir -e .

CPP_EXTENSION_PATH=$(find "$BUILD_DIR" -name "hegemonikon_py*.so" -print -quit)

if [ -z "$CPP_EXTENSION_PATH" ]; then
    echo "ERROR: Could not find compiled C++ extension (hegemonikon_py.so) in the build directory. Build failed."
    deactivate
    exit 1
fi
echo "Found C++ extension at: $CPP_EXTENSION_PATH"

pyinstaller --noconfirm \
            --onefile \
            --name "api" \
            --distpath "$DIST_DIR" \
            --add-binary "$CPP_EXTENSION_PATH:ataraxai" \
            --hidden-import "ataraxai.hegemonikon_py" \
            --exclude-module pytest \
            --exclude-module mypy \
            --exclude-module ruff \
            api.py

ARTIFACT_PATH="$DIST_DIR/api"
strip "$ARTIFACT_PATH"
mkdir -p "$TAURI_RESOURCE_DIR"
cp "$ARTIFACT_PATH" "$TAURI_RESOURCE_DIR"/

deactivate

