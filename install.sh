#!/bin/bash
set -e

# --- Initial Configuration ---
CLEAN=0
USE_CONDA=0
USE_CUDA=0
CUDA_ARCH="native"
SETUP_ARGS="" 
CMAKE_ARGS_STR="" 

clear
echo "Starting Atarax-AI Installation..."

for arg in "$@"; do
    case $arg in
        --clean) CLEAN=1 ;;
        --use-cuda)
            USE_CUDA=1
            SETUP_ARGS+=" --use-cuda"
            ;;
        --cuda-arch=*)
            CUDA_ARCH="${arg#*=}"
            SETUP_ARGS+=" --cuda-arch=${CUDA_ARCH}"
            ;;
        --use-conda)
            USE_CONDA=1
            SETUP_ARGS+=" --use-conda"
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done


if [[ $USE_CONDA -eq 1 ]]; then
    if [[ -z "$CONDA_PREFIX" ]]; then
        echo "Error: --use-conda specified, but CONDA_PREFIX is not set. Please activate your conda environment first."
        exit 1
    fi
    echo "[+] Configuring for Conda environment at $CONDA_PREFIX"
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi


if [[ $CLEAN -eq 1 ]]; then
    echo "[+] Cleaning previous build artifacts..."
    rm -rf build ataraxai_assistant.egg-info dist _skbuild
    rm -f ataraxai/*.so
    [ -f clean.sh ] && ./clean.sh
fi

echo "[+] Uninstalling any previous version..."
pip uninstall ataraxai_assistant -y || true



if [ -f "setup_third_party.sh" ]; then
    echo "[+] Setting up third-party libraries..."
    ./setup_third_party.sh ${SETUP_ARGS}
fi



if [[ $USE_CUDA -eq 1 ]]; then
    echo "[+] Configuring build for CUDA=ON"
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=ON"
else
    echo "[+] Configuring build for CUDA=OFF"
    CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=OFF"
fi
export CMAKE_ARGS="${CMAKE_ARGS_STR}"
echo "[i] CMake arguments for pip: ${CMAKE_ARGS}"

# echo " Running CMake..."
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release

echo "[+] Running pip install..."
pip install -e . --verbose


echo "[+] Verifying installation..."
python -c "from ataraxai import core_ai_py; print('[SUCCESS] Atarax-AI installed and core module is importable!')"