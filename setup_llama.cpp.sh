git clone https://github.com/ggml-org/llama.cpp.git
git clone https://github.com/ggml-org/whisper.cpp.git
git clone https://github.com/opencv/opencv.git
cd llama.cpp
rm -rf build CMakeCache.txt CMakeFiles
conda install conda-forge::curl
conda install -n ataraxai libcurl libssh openldap
conda install libcurl libidn2 libpsl libssh openldap
mkdir build
cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=$CONDA_ENV_PATH \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURL_LIBRARY=$CONDA_ENV_PATH/lib/libcurl.so \
  -DCURL_INCLUDE_DIR=$CONDA_ENV_PATH/include
cd ../..