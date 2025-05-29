#!/bin/bash

set -e

rm -rf build

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

# Define and export the AtaraxIA project root
export ATARAXIA_PATH="$(pwd)"
echo "Building AtaraxIA from $ATARAXIA_PATH"

# Optional: setup third-party dependencies
if [ -f "setup_third_party.sh" ]; then
    echo "Setting up third-party libraries..."
    ./setup_third_party.sh
fi

# Configure and build the project
echo " Running CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Done!
echo "AtaraxIA built successfully."
echo "Run it with: export ATARAXIA_PATH=$('pwd')"
echo "Then ./build/bin/ataraxia"
