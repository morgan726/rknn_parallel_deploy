set -e

GCC_COMPILER=aarch64-linux-gnu
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake .. -DCMAKE_SYSTEM_NAME=Linux
make -j8


