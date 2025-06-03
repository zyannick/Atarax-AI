#!/bin/bash

set -e

CUDA_ARCH="native"
LLAMA_TAG="tags/b5581"
WHISPER_TAG="tags/v1.7.5"

for arg in "$@"; do
    case $arg in
        --cuda-arch=*) CUDA_ARCH="${arg#*=}" ;;
        --llama-tag=*) LLAMA_TAG="${arg#*=}" ;;
        --whisper-tag=*) WHISPER_TAG="${arg#*=}" ;;
    esac
done

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

rm -rf build

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_SRC_DIR="$PROJECT_ROOT/ataraxai/core_ai/third_party"
THIRD_PARTY_INSTALL_DIR="$PROJECT_ROOT/build/third_party_install"

LLAMA_CPP_SRC_DIR="$THIRD_PARTY_SRC_DIR/llama.cpp"
LLAMA_CPP_BUILD_DIR="$PROJECT_ROOT/build/cmake_build/llama_cpp"
LLAMA_CPP_INSTALL_DIR="$THIRD_PARTY_INSTALL_DIR/llama"

WHISPER_CPP_SRC_DIR="$THIRD_PARTY_SRC_DIR/whisper.cpp"
WHISPER_CPP_BUILD_DIR="$PROJECT_ROOT/build/cmake_build/whisper_cpp"
WHISPER_CPP_INSTALL_DIR="$THIRD_PARTY_INSTALL_DIR/whisper"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p "$THIRD_PARTY_SRC_DIR"
mkdir -p "$LLAMA_CPP_BUILD_DIR" "$WHISPER_CPP_BUILD_DIR"
mkdir -p "$LLAMA_CPP_INSTALL_DIR" "$WHISPER_CPP_INSTALL_DIR"

echo "Cleaning old sources..."
rm -rf $THIRD_PARTY_SRC_DIR/llama.cpp $THIRD_PARTY_SRC_DIR/whisper.cpp $THIRD_PARTY_SRC_DIR/boost

echo "Cloning llama.cpp..."
if [ ! -d "$THIRD_PARTY_SRC_DIR/llama.cpp/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$THIRD_PARTY_SRC_DIR/llama.cpp"
fi
cd $THIRD_PARTY_SRC_DIR/llama.cpp && git checkout "$LLAMA_TAG"
echo "Cloning whisper.cpp..."
if [ ! -d "$THIRD_PARTY_SRC_DIR/whisper.cpp/.git" ]; then
    git clone https://github.com/ggml-org/whisper.cpp.git "$THIRD_PARTY_SRC_DIR/whisper.cpp"
fi
cd $THIRD_PARTY_SRC_DIR/whisper.cpp && git checkout "$WHISPER_TAG"

cmake -S "$LLAMA_CPP_SRC_DIR" -B "$LLAMA_CPP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX="$LLAMA_CPP_INSTALL_DIR" \
    -DLLAMA_CUDA=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON  
    # -DLLAMA_CUDA_ARCH="$CUDA_ARCH" 

cmake --build "$LLAMA_CPP_BUILD_DIR" --config Release -j4
cmake --install "$LLAMA_CPP_BUILD_DIR" --config Release

cmake -S "$WHISPER_CPP_SRC_DIR" -B "$WHISPER_CPP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_TESTS=ON -DWHISPER_BUILD_EXAMPLES=ON \
    -DCMAKE_INSTALL_PREFIX="$WHISPER_CPP_INSTALL_DIR" \
    -DWHISPER_CUDA=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON 
    # -DWHISPER_CUDA_ARCH="$CUDA_ARCH" 
cmake --build "$WHISPER_CPP_BUILD_DIR" --config Release -j4
cmake --install "$WHISPER_CPP_BUILD_DIR" --config Release
