#!/bin/bash
set -e

clear 

CLEAN=0
CUDA_ARCH="native"

# Parse CLI args
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=1 ;;
        --cuda-arch=*) CUDA_ARCH="${arg#*=}" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [[ $CLEAN -eq 1 ]]; then
    echo "[+] Cleaning build artifacts"
    rm -rf build ataraxai_assistant.egg-info dist *.so _skbuild
    ./clean.sh || true
fi

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH

./clean.sh

pip uninstall ataraxai_assistant -y || true

export ATARAXIA_PATH="$(pwd)"
echo "Setting up AtaraxIA from $ATARAXIA_PATH"

if [ -f "setup_third_party.sh" ]; then
    echo "Setting up third-party libraries..."
    ./setup_third_party.sh
fi

echo " Running CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release


pip install -e .  -v     

python -c "from ataraxai import core_ai_py; print('AtaraxAI installed successfully!')"