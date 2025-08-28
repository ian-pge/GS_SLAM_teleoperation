#!/usr/bin/env bash

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

sudo chown -R zed:zed /usr/local/zed

export TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0"

# opencv4
echo "Building OpenCV ..."
cmake -B third_party/opencv/build -G Ninja \
      -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_CUFFT=ON \
      -DWITH_CUBLAS=ON \
      -DWITH_NVCUVENC=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DWITH_NVCUVID=ON \
      -DBUILD_TIFF=ON \
      -DBUILD_ZLIB=ON \
      -DBUILD_JASPER=ON \
      -DBUILD_CCALIB=ON \
      -DBUILD_JPEG=ON \
      -DWITH_FFMPEG=ON \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_EXTRA_MODULES_PATH=$SCRIPT_DIR/third_party/opencv_contrib/modules \
      -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR/third_party/install/opencv \
      third_party/opencv
cmake --build third_party/opencv/build
cmake --install third_party/opencv/build

# DBoW2
cmake -B third_party/ORB-SLAM3/Thirdparty/DBoW2/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=$SCRIPT_DIR/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3/Thirdparty/DBoW2
cmake --build third_party/ORB-SLAM3/Thirdparty/DBoW2/build

# g2o
cmake -B third_party/ORB-SLAM3/Thirdparty/g2o/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/g2o
cmake --build third_party/ORB-SLAM3/Thirdparty/g2o/build

# Sophus
cmake -B third_party/ORB-SLAM3/Thirdparty/Sophus/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/Sophus
cmake --build third_party/ORB-SLAM3/Thirdparty/Sophus/build
# Don't forget to add
#include <sstream> in /workspaces/cartgs2/third_party/ORB-SLAM3/Thirdparty/Sophus/sophus/formatstring.hpp
#include <boost/archive/basic_archive.hpp> in KeyFrameDatabase.h

# ORB-SLAM3
echo "Uncompress vocabulary ..."
tar -xf third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz \
    -C third_party/ORB-SLAM3/Vocabulary

cmake -B third_party/ORB-SLAM3/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=$SCRIPT_DIR/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3
cmake --build third_party/ORB-SLAM3/build

# CaRtGS
echo "Building CaRtGS ..."
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=$SCRIPT_DIR/third_party/install/opencv/lib/cmake/opencv4 \
      -DTorch_DIR=/usr/local/libtorch/share/cmake/Torch
cmake --build build

# Change the python packages owner for colcon
sudo chown -R zed:zed /usr/local/lib/python3.10/dist-packages

# Prepare ROS 2 workspace
if ! grep -qF "source /opt/ros/$ROS_DISTRO/setup.bash" /home/zed/.bashrc; then
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/zed/.bashrc
fi

source /opt/ros/$ROS_DISTRO/setup.bash

# Build ros2_ws
echo "Building ros2_ws ..."
sudo apt-get update -y
rosdep update
sudo rosdep install \
  --from-paths "$SCRIPT_DIR/ros2_ws/src" \
  --ignore-src \
  --rosdistro "$ROS_DISTRO" \
  -r -y
colcon build \
      --build-base $SCRIPT_DIR/ros2_ws/build \
      --install-base $SCRIPT_DIR/ros2_ws/install \
      --base-paths $SCRIPT_DIR/ros2_ws/src \
      --cmake-args \
        -DCARTGS_ROOT_DIR="$SCRIPT_DIR" \
        -DORB_SLAM3_ROOT_DIR="$SCRIPT_DIR/third_party/ORB-SLAM3" \
        -DOpenCV_DIR="$SCRIPT_DIR/third_party/install/opencv/lib/cmake/opencv4" \
        -DTorch_DIR=/usr/local/libtorch/share/cmake/Torch \
        -DCMAKE_INSTALL_RPATH="$SCRIPT_DIR/cartgs_ros_rolling/third_party/install/opencv/lib" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE


if ! grep -qF "source $SCRIPT_DIR/ros2_ws/install/setup.bash" /home/zed/.bashrc; then
    echo "source $SCRIPT_DIR/ros2_ws/install/setup.bash" >> /home/zed/.bashrc
fi

source /opt/ros/$ROS_DISTRO/setup.bash
source $SCRIPT_DIR/ros2_ws/install/setup.bash
