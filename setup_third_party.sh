#!/bin/bash

set -e 

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

# Directories
ROOT_DIR=$(pwd) 
THIRD_PARTY_SRC_DIR=$ROOT_DIR/ataraxai/third_party
THIRD_PARTY_BUILD_DIR=$ROOT_DIR/build/third_party
ATARAXAI_TEMP_DIR=$ROOT_DIR/temp


BOOST_VERSION="1.88.0" 
BOOST_VERSION_UNDERSCORE="1_88_0"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p $THIRD_PARTY_SRC_DIR
mkdir -p $THIRD_PARTY_BUILD_DIR
mkdir -p $ATARAXAI_TEMP_DIR


echo "Cleaning old sources..."
rm -rf $THIRD_PARTY_SRC_DIR/llama.cpp $THIRD_PARTY_SRC_DIR/whisper.cpp $THIRD_PARTY_SRC_DIR/boost

echo "Cloning llama.cpp..."
git clone https://github.com/ggml-org/llama.cpp.git $THIRD_PARTY_SRC_DIR/llama.cpp
echo "Cloning whisper.cpp..."
git clone https://github.com/ggml-org/whisper.cpp.git $THIRD_PARTY_SRC_DIR/whisper.cpp
cd $ROOT_DIR 

# Build llama.cpp
cmake -S $THIRD_PARTY_SRC_DIR/llama.cpp -B $THIRD_PARTY_BUILD_DIR/llama.cpp -DOPENSSL_ROOT_DIR=$CONDA_PREFIX -DLLAMA_CUDA=ON  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build $THIRD_PARTY_BUILD_DIR/llama.cpp --config Release -j4

# Verify llama binary
if [ ! -f $THIRD_PARTY_BUILD_DIR/llama.cpp/bin/llama-quantize ]; then
    echo "Llama.cpp build failed."
    exit 1
fi

# Build whisper.cpp
cmake -S $THIRD_PARTY_SRC_DIR/whisper.cpp -B $THIRD_PARTY_BUILD_DIR/whisper.cpp  -DWHISPER_CUDA=ON
cmake --build $THIRD_PARTY_BUILD_DIR/whisper.cpp --config Release -j4

# Verify whisper binary
if [ ! -f $THIRD_PARTY_BUILD_DIR/whisper.cpp/bin/whisper-cli ]; then
    echo "Whisper.cpp build failed."
    exit 1
fi

