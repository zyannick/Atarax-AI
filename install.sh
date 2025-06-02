#!/bin/bash
set -e

rm -rf build ataraxai_assistant.egg-info dist *.so _skbuild

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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

echo "Check if ataraxai is installed"
python -c "import ataraxai"