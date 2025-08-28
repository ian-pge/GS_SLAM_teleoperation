# -manualLicenseFile /workspace/Unity_lic.ulf -logFile /dev/stdout
ARG TARGETARCH=amd64
FROM unityci/editor:ubuntu-6000.0.44f1-base-3.1.0 AS base

# Common packages needed for the keyring steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

###############################################################################
# 1a) Base - AMD64
###############################################################################
FROM base AS base-amd64

ENV NVARCH=x86_64

# Note: This big environment variable string is straight from the
# official CUDA Dockerfiles. It's what ensures driver compatibility.

ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566"

# Matching versions from the official NVIDIA docs
ENV NV_CUDA_CUDART_VERSION=12.8.90-1
ENV CUDA_VERSION=12.8.1

###############################################################################
# 1b) Base - ARM64
###############################################################################
FROM base AS base-arm64

ENV NVARCH=sbsa

ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566"
ENV NV_CUDA_CUDART_VERSION=12.8.90-1
ENV CUDA_VERSION=12.8.1

###############################################################################
# 1c) Base (common) - Install CUDA keyring, etc.
###############################################################################
FROM base-${TARGETARCH} AS base-final

# We add the CUDA keyring + minimal runtime items
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

# For libraries in the cuda-compat-* package
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-8=${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
# COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

###############################################################################
# 2) Runtime libraries stage (CUDA libraries, NPP, cuBLAS, NCCL, etc.)
###############################################################################
FROM base-final AS runtime-libs

ENV NV_CUDA_LIB_VERSION=12.8.1-1
ENV NV_NVTX_VERSION=12.8.90-1
ENV NV_LIBNPP_VERSION=12.3.3.100-1
ENV NV_LIBNPP_PACKAGE="libnpp-12-8=${NV_LIBNPP_VERSION}"
ENV NV_LIBCUSPARSE_VERSION=12.5.8.93-1

ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8
ENV NV_LIBCUBLAS_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_PACKAGE="${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}"

ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_PACKAGE="${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-8=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-8=${NV_NVTX_VERSION} \
    libcusparse-12-8=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages.
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# Add entrypoint items (optional)
# If you have an entrypoint script from the original Dockerfiles:
# COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
# COPY nvidia_entrypoint.sh /opt/nvidia/
# ENV NVIDIA_PRODUCT_NAME="CUDA"
# ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

###############################################################################
# 3) Development libraries stage (compiler, dev libs, nsight, etc.)
###############################################################################
FROM runtime-libs AS devel

ENV NV_CUDA_CUDART_DEV_VERSION=12.8.90-1
ENV NV_NVML_DEV_VERSION=12.8.90-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1
ENV NV_LIBNPP_DEV_VERSION=12.3.3.100-1
ENV NV_LIBNPP_DEV_PACKAGE="libnpp-dev-12-8=${NV_LIBNPP_DEV_VERSION}"

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8
ENV NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE="${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}"

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="cuda-nsight-compute-12-8=${NV_CUDA_NSIGHT_COMPUTE_VERSION}"

ENV NV_NVPROF_VERSION=12.8.90-1
ENV NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-8=${NV_NVPROF_VERSION}"

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_DEV_PACKAGE="${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-8=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-8=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-8=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Prevent cublas/nccl auto-upgrades
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

###############################################################################
# 4) Final stage: Add cuDNN
###############################################################################
FROM devel AS final

# cuDNN environment variables
ENV NV_CUDNN_VERSION=9.8.0.87-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE="${NV_CUDNN_PACKAGE_NAME}=${NV_CUDNN_VERSION}"
ENV NV_CUDNN_PACKAGE_DEV="libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}"

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>" \
    com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# This final image now contains:
#   1) CUDA Base
#   2) CUDA Runtime Libraries
#   3) CUDA Development Libraries
#   4) cuDNN Libraries
# If you had an entrypoint, copy or enable it here.
# --------------------------------------------------------------------------
# If you want a default entrypoint script, you can uncomment something like:
# COPY nvidia_entrypoint.sh /opt/nvidia/
# ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

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
