#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sudo chown -R zed:zed /usr/local/zed

# opencv4
echo "Building OpenCV ..."
cmake -B third_party/opencv/build -G Ninja \
      -DCMAKE_CUDA_ARCHITECTURES="86" \
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
# if ! grep -qF "source /opt/ros/$ROS_DISTRO/setup.zsh" ~/.zshrc.local; then
#     echo "source /opt/ros/$ROS_DISTRO/setup.zsh" >> ~/.zshrc.local
# fi
# if ! grep -qF "source $SCRIPT_DIR/ros2_ws/install/setup.zsh" ~/.zshrc.local; then
#     echo "source $SCRIPT_DIR/ros2_ws/install/setup.zsh" >> ~/.zshrc.local
# fi

source /opt/ros/$ROS_DISTRO/setup.bash

git init $SCRIPT_DIR/ros2_ws/src

# Clone and build ZED ROS 2 Wrapper
echo "Setting up ZED ROS 2 Wrapper ..."
git clone https://github.com/stereolabs/zed-ros2-wrapper.git "$SCRIPT_DIR/ros2_ws/src/zed-ros2-wrapper"
sudo apt-get update -y
rosdep update # add on foxy --include-eol-distros
rosdep install --from-paths $SCRIPT_DIR/ros2_ws/src/zed-ros2-wrapper --ignore-src -r -y

# Clone open_cv_bridge from source to be able to use same OpenCV version than ORB-SLAM3
echo "Downloading open_cv_bridge ..."
git clone -b humble https://github.com/ros-perception/vision_opencv.git "$SCRIPT_DIR/ros2_ws/src/vision_opencv"

# Clone Unity ROS2 Bridge
echo "Downloading Unity ROS2 Bridge ..."
git clone -b main-ros2 https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git "$SCRIPT_DIR/ros2_ws/src/ROS-TCP-Endpoint"

# Build ros2_ws
echo "Building ros2_ws ..."
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

source $SCRIPT_DIR/ros2_ws/install/setup.bash
