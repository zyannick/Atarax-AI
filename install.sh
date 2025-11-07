#!/bin/bash
set -e

CLEAN=0
CLEAN_CCACHE=0
USE_CONDA=0
USE_CUDA=0
CUDA_ARCH=""
CMAKE_ARGS_STR=""
ONLY_CPP=0
USE_UV=0

for arg in "$@"; do
	case $arg in
	--clean) CLEAN=1 ;;
	--use-cuda)
		USE_CUDA=1
		;;
	--clean-ccache)
		CLEAN_CCACHE=1
		echo "[+] Cleaning ccache..."
		ccache -C || echo "ccache not installed or clean failed."
		;;
	--cuda-arch=*)
		CUDA_ARCH="${arg#*=}"
		;;
	--only-cpp)
		ONLY_CPP=1
		echo "[+] Only building C++ components."
		;;
	--use-conda)
		USE_CONDA=1
		;;
	--use-uv)
		USE_UV=1
		;;
	*)
		echo "Unknown option: $arg"
		exit 1
		;;
	esac
done

if [[ $CLEAN -eq 1 ]]; then
	rm -rf build ataraxai.egg-info dist _skbuild
	rm -f ataraxai/*.so
	[ -f clean.sh ] && ./clean.sh
fi

if [[ $USE_UV -eq 1 ]]; then
	if ! command -v uv &>/dev/null; then
		echo "Error: uv is not installed or not in PATH. Please install it first."
		exit 1
	fi
	uv venv .venv --clear
	source .venv/bin/activate
	uv pip install cmake scikit-build ninja
fi

if [[ $USE_CONDA -eq 1 ]]; then
	if [[ -z $CONDA_PREFIX ]]; then
		echo "Error: --use-conda specified, but CONDA_PREFIX is not set. Please activate your conda environment first."
		exit 1
	fi
	echo "[+] Configuring for Conda environment at $CONDA_PREFIX"
	export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
	export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
	export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

echo "[+] Uninstalling any previous version..."
if [[ $USE_UV -eq 1 ]]; then
	uv pip uninstall ataraxai || true
else
	pip uninstall ataraxai -y || true
fi

echo "[i] Third-party dependencies will be handled by CMake (FetchContent)."


if [[ $USE_CUDA -eq 1 ]]; then
	echo "[+] Configuring build for CUDA=ON"
	CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=ON"
	if [ -n "$CUDA_ARCH" ]; then
		CMAKE_ARGS_STR+=" -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
	fi
else
	echo "[+] Configuring build for CUDA=OFF"
	CMAKE_ARGS_STR="-DATARAXAI_USE_CUDA=OFF"
fi

CMAKE_ARGS_STR+=" -DBUILD_TESTING=ON"

# Set CMAKE_ARGS for pip/scikit-build to pick up
export CMAKE_ARGS="${CMAKE_ARGS_STR}"
echo "[i] CMake arguments for pip: ${CMAKE_ARGS}"

# We still run cmake manually to build the test targets
PYTHON_EXECUTABLE=$(which python3)
if [ -z "$PYTHON_EXECUTABLE" ]; then
	echo "Error: python3 not found in PATH."
	exit 1
fi
PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "[+] Configuring CMake..."
cmake -S . -B build \
	-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
	-DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
	${CMAKE_ARGS_STR}

echo "[+] Building C++ targets (including tests)..."
cmake --build build --config Release -- -j $(nproc)
cmake --build build --target hegemonikon_tests --config Release

if [[ $ONLY_CPP -eq 1 ]]; then
	echo "[+] C++ build complete. Exiting due to --only-cpp flag."
	[[ $USE_UV -eq 1 ]] && deactivate
	exit 0
fi

echo "[+] Installing Python package..."
if [[ $USE_UV -eq 1 ]]; then
	uv pip install -e .
else
	python3 -m pip install -e .
fi

echo "[+] Verifying installation..."
python3 -c "from ataraxai import hegemonikon_py; print('[SUCCESS] Atarax-AI installed and core module is importable!')"
if [[ $USE_UV -eq 1 ]]; then
	deactivate
	echo "[+] Build complete. Virtual environment deactivated."
fi