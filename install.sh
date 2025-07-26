#!/bin/bash
set -e

CLEAN=0
CLEAN_CCACHE=0
USE_CONDA=0
USE_CUDA=0
CUDA_ARCH="native"
SETUP_ARGS=""
CMAKE_ARGS_STR=""
ONLY_CPP=0

echo "Starting Atarax-AI Installation..."

for arg in "$@"; do
    case $arg in
    --clean) CLEAN=1 ;;
    --use-cuda)
        USE_CUDA=1
        SETUP_ARGS+=" --use-cuda"
        ;;
    --clean-ccache)
        CLEAN_CCACHE=1
        echo "[+] Cleaning ccache..."
        ccache -C || echo "ccache not installed or clean failed."
        ;;
    --cuda-arch=*)
        CUDA_ARCH="${arg#*=}"
        SETUP_ARGS+=" --cuda-arch=${CUDA_ARCH}"
        ;;
    --only-cpp)
        ONLY_CPP=1
        SETUP_ARGS+=" --only-cpp"
        echo "[+] Only building C++ components."
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
    rm -rf build ataraxai.egg-info dist _skbuild
    rm -f ataraxai/*.so
    [ -f clean.sh ] && ./clean.sh
fi

echo "[+] Uninstalling any previous version..."
pip uninstall ataraxai -y || true

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

CMAKE_ARGS_STR+=" -DBUILD_TESTING=ON -DPYTHON_EXECUTABLE=$(which python3)"

export CMAKE_ARGS="${CMAKE_ARGS_STR}"
echo "[i] CMake arguments for pip: ${CMAKE_ARGS}"

PYTHON_EXECUTABLE=$(which python3)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "Error: python3 not found in PATH. Please ensure Python 3 is installed and available."
    exit 1
fi

PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo " Running CMake Configuration..."
cmake -S . -B build \
    -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}

echo " Building Main Dependencies..."
cmake --build build --config Release


echo " Building C++ Tests..."
cmake --build build --target hegemonikon_tests

if [[ $ONLY_CPP -eq 1 ]]; then
    echo "[+] C++ build complete. Exiting due to --only-cpp flag."
    exit 0
fi

echo "[+] Running pip install to build Python extension..."
python3 -m pip install -e .

echo "[+] Verifying installation..."
python3 -c "from ataraxai import hegemonikon_py; print('[SUCCESS] Atarax-AI installed and core module is importable!')"
