FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN sed -i 's|archive.ubuntu.com|mirror.math.princeton.edu/pub/ubuntu|' \
  /etc/apt/sources.list && \
  apt-get update -o Acquire::Retries=3

#-------------------------------------------------------------------------------------------------
# SETUP THE USER

# Install necessary dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  sudo \
  xz-utils \
  ca-certificates \
  git \
  && \
  rm -rf /var/lib/apt/lists/*

ARG USERNAME=zed
ARG USER_UID=1000
ARG USER_GID=1000

RUN \
  # --- 1) GROUP: Check if GID=1000 exists ---
  if getent group ${USER_GID} >/dev/null; then \
  # If a group with GID=1000 exists but has a different name, rename it.
  old_group_name="$(getent group ${USER_GID} | cut -d: -f1)"; \
  if [ "$old_group_name" != "$USERNAME" ]; then \
  groupmod -n "$USERNAME" "$old_group_name"; \
  fi; \
  else \
  # Otherwise create a fresh group for $USERNAME
  groupadd --gid "$USER_GID" "$USERNAME"; \
  fi \
  \
  # --- 2) USER: Check if UID=1000 exists ---
  && if getent passwd ${USER_UID} >/dev/null; then \
  # If a user with UID=1000 exists but has a different name, rename it.
  old_user_name="$(getent passwd ${USER_UID} | cut -d: -f1)"; \
  if [ "$old_user_name" != "$USERNAME" ]; then \
  usermod -l "$USERNAME" -d /home/"$USERNAME" -m "$old_user_name"; \
  fi; \
  else \
  # Otherwise create a fresh user for $USERNAME
  useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME"; \
  fi \
  \
  # --- 3) Give the user passwordless sudo ---
  && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

#-------------------------------------------------------------------------------------------------

ARG UBUNTU_RELEASE_YEAR=22
ARG UBUNTU_MINOR=04
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=1
ARG CUDA_PATCH=0
ARG ZED_SDK_MAJOR=5
ARG ZED_SDK_MINOR=0
ARG ZED_SDK_PATCH=1
#go check the last veresion

ARG ROS2_DIST=humble

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility

# Disable apt-get warnings
RUN apt-get update || true && apt-get install -y --no-install-recommends apt-utils dialog

# ZED SDK link
ENV ZED_SDK_URL="https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/cu${CUDA_MAJOR}/ubuntu${UBUNTU_RELEASE_YEAR}"

ENV TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
  apt-get update && \
  apt-get install --yes lsb-release wget less udev sudo build-essential cmake python3 python3-dev python3-pip python3-wheel git jq libpq-dev zstd usbutils
# libopencv-dev
############ Install ROS2 ############

# Set and Check Locale
RUN apt-get update || true && \
  apt-get install --no-install-recommends -y locales && \
  locale-gen en_US en_US.UTF-8 && \
  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
  export LANG=en_US.UTF-8 && \
  locale  # verify settings

# Setup Sources
RUN apt-get update || true && \
  apt-get install --no-install-recommends -y software-properties-common && \
  add-apt-repository universe && \
  apt-get install -y curl && \
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update || true && \
  apt-get install --no-install-recommends -y \
  ros-${ROS2_DIST}-ros-base \
  # ros-${ROS2_DIST}-ament-cmake-clang-format \
  # ros-${ROS2_DIST}-image-transport \
  # ros-${ROS2_DIST}-image-transport-plugins \
  # ros-${ROS2_DIST}-diagnostic-updater \
  # ros-${ROS2_DIST}-xacro \
  python3-flake8-docstrings \
  python3-pip \
  python3-pytest-cov \
  ros-dev-tools && \
  pip3 install \
  argcomplete \
  numpy \
  empy \
  lark

# Initialize rosdep
RUN rosdep init && rosdep update

# Setup the ZED SDK
RUN apt-get update -y || true && \
  apt-get install --no-install-recommends dialog bash-completion lsb-release wget less udev sudo  build-essential cmake zstd python3 python3-pip libpng-dev libgomp1 -y && \
  wget -O ZED_SDK_Linux_Ubuntu.run ${ZED_SDK_URL} && \
  chmod +x ZED_SDK_Linux_Ubuntu.run && \
  ./ZED_SDK_Linux_Ubuntu.run -- silent skip_cuda && \
  ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so && \
  rm ZED_SDK_Linux_Ubuntu.run

# Install other dependencies
RUN apt-get update && \
  apt-get install -y \
  git \
  build-essential \
  sudo \
  libeigen3-dev \
  libjsoncpp-dev \
  libopengl-dev \
  mesa-utils \
  libglfw3-dev \
  libglm-dev \
  python3-pip \
  python3-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  curl \
  zip \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libswresample-dev \
  libssl-dev \
  nlohmann-json3-dev \
  unzip \
  tar \
  python-is-python3 \
  libboost-all-dev \
  ninja-build

# Install libtorch
RUN wget  https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip \
  -O /tmp/libtorch.zip && \
  unzip /tmp/libtorch.zip -d /usr/local && \
  rm /tmp/libtorch.zip

# Setting up environment variables
ENV PATH="/usr/local/bin:${PATH}"
ENV ROS_DISTRO ${ROS2_DIST}

# Clean up
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*
