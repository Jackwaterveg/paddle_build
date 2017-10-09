#!/bin/bash

set -xe

PROJ_ROOT=$PWD
SUFFIX=_ios

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

cd $BUILD_ROOT
cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
      -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
      -DCMAKE_SYSTEM_NAME=iOS \
      -DIOS_PLATFORM=OS \
      -DCMAKE_OSX_ARCHITECTURES="arm64" \
			-DWITH_C_API=ON \
      -DUSE_EIGEN_FOR_BLAS=ON \
			-DWITH_TESTING=ON \
			-DWITH_SWIG_PY=OFF \
      -DWITH_STYLE_CHECK=OFF \
			-DCMAKE_BUILD_TYPE=Release \
			$SOURCES_ROOT

cd $PROJ_ROOT