#!/bin/bash 

set -e

function installEigenpy () {
  # install eigenpy
  uv pip install numpy
  if [ -d "eigenpy" ]; then
      rm -rf eigenpy
  fi
  git clone https://github.com/stack-of-tasks/eigenpy.git
  cd eigenpy
  mkdir build && cd build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
    -DPYTHON_EXECUTABLE=$(which python)

  make -j4 
  make install
  cd ../../
}

function installCasADi(){
  # install casadi
  if [ -d "casadi" ]; then
      rm -rf casadi
  fi
  git clone https://github.com/casadi/casadi.git
  cd casadi
  mkdir build && cd build
  sudo apt install swig liblapack-dev libblas-dev -y
  cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python) \
      -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
      -DWITH_BUILD_REQUIRED=ON \
      -DWITH_BUILD_IPOPT=ON \
      -DWITH_IPOPT=ON \
      -DWITH_PYTHON=ON

  make -j4
  make install
  cd ../../
}

function installPinocchio(){
  sudo apt install liburdfdom-headers-dev liburdfdom-dev -y
  # install pinocchio
  if [ -d "pinocchio" ]; then
      rm -rf pinocchio
  fi
  # git clone https://github.com/stack-of-tasks/pinocchio.git
  git clone git@github.com:stack-of-tasks/pinocchio.git
  cd pinocchio
  # high version cannot cmake example-robot-data
  git reset --hard bb5658416724a36d5e8d2fb6c65614f39796f7f1
  mkdir build && cd build
  # cd build

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I${VIRTUAL_ENV}/include -L${VIRTUAL_ENV}/lib" \
    -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
    -DBUILD_WITH_CASADI_SUPPORT=ON \
    -DCMAKE_PREFIX_PATH="$VIRTUAL_ENV" \
    -DCMAKE_INCLUDE_PATH="${VIRTUAL_ENV}/include" \
    -Dcasadi_DIR="${VIRTUAL_ENV}/lib/cmake/casadi" \
    -DCMAKE_LIBRARY_PATH="${VIRTUAL_ENV}/lib"

  make -j4
  make install
}

source ../.venv/bin/activate
inscdtallEigenpy
installCasADi
installPinocchio