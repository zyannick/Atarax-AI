#!/bin/bash

set -e 

# Directories
THIRD_PARTY_SRC_DIR=ataraxai/third_party
THIRD_PARTY_BUILD_DIR=build/third_party

# Clean old sources
rm -rf $THIRD_PARTY_SRC_DIR/llama.cpp $THIRD_PARTY_SRC_DIR/whisper.cpp

# Clone dependencies
git clone https://github.com/ggml-org/llama.cpp.git $THIRD_PARTY_SRC_DIR/llama.cpp
git clone https://github.com/ggml-org/whisper.cpp.git $THIRD_PARTY_SRC_DIR/whisper.cpp

# Build llama.cpp
cmake -S $THIRD_PARTY_SRC_DIR/llama.cpp -B $THIRD_PARTY_BUILD_DIR/llama.cpp
cmake --build $THIRD_PARTY_BUILD_DIR/llama.cpp --config Release -j4

# Verify llama binary
if [ ! -f $THIRD_PARTY_BUILD_DIR/llama.cpp/bin/llama-quantize ]; then
    echo "Llama.cpp build failed."
    exit 1
fi

# Build whisper.cpp
cmake -S $THIRD_PARTY_SRC_DIR/whisper.cpp -B $THIRD_PARTY_BUILD_DIR/whisper.cpp
cmake --build $THIRD_PARTY_BUILD_DIR/whisper.cpp --config Release -j4

# Verify whisper binary
if [ ! -f $THIRD_PARTY_BUILD_DIR/whisper.cpp/bin/whisper-cli ]; then
    echo "Whisper.cpp build failed."
    exit 1
fi

echo "Third-party dependencies built successfully."
