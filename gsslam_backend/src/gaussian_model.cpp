/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM, and modified by Dapeng Feng in 2024, as part of
 * CaRtGS.
 */

#include "include/gaussian_model.h"

#include "include/gaussian_rasterizer.h"

GaussianModel::GaussianModel(const int sh_degree)
    : active_sh_degree_(0),
      spatial_lr_scale_(0.0),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1000000) {
  this->max_sh_degree_ = sh_degree;

  // Device
  if (torch::cuda::is_available())
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

  // Allocate fixed-size tensors initialized with zeros

  // Allocate fixed-size cudaMalloc tensors for IPC
  size_t max_num_gaussians = MAX_NUM_POINTS;

  size_t xyz_num_elements = max_num_gaussians * 3;
  size_t features_dc_num_elements = max_num_gaussians * 1 * 3;
  size_t scaling_num_elements = max_num_gaussians * 3;
  size_t rotation_num_elements = max_num_gaussians * 4;
  size_t opacity_num_elements = max_num_gaussians * 1;

  // Allocate device memory using cudaMalloc
  cudaMalloc(&xyz_ipc_, xyz_num_elements * sizeof(float));
  cudaMalloc(&features_dc_ipc_, features_dc_num_elements * sizeof(float));
  cudaMalloc(&scaling_ipc_, scaling_num_elements * sizeof(float));
  cudaMalloc(&rotation_ipc_, rotation_num_elements * sizeof(float));
  cudaMalloc(&opacity_ipc_, opacity_num_elements * sizeof(float));

  // Initialize
  cudaMemset(opacity_ipc_, 0, opacity_num_elements * sizeof(float));

  // Get IPC handles
  cudaIpcGetMemHandle(&xyz_ipc_handle_, xyz_ipc_);
  cudaIpcGetMemHandle(&features_dc_ipc_handle_, features_dc_ipc_);
  cudaIpcGetMemHandle(&scaling_ipc_handle_, scaling_ipc_);
  cudaIpcGetMemHandle(&rotation_ipc_handle_, rotation_ipc_);
  cudaIpcGetMemHandle(&opacity_ipc_handle_, opacity_ipc_);

  // Collect tensors to share
  tensors_to_share_["xyz"] = std::make_pair(xyz_ipc_, xyz_ipc_handle_);
  tensors_to_share_["features_dc"] = std::make_pair(features_dc_ipc_, features_dc_ipc_handle_);
  tensors_to_share_["scaling"] = std::make_pair(scaling_ipc_, scaling_ipc_handle_);
  tensors_to_share_["rotation"] = std::make_pair(rotation_ipc_, rotation_ipc_handle_);
  tensors_to_share_["opacity"] = std::make_pair(opacity_ipc_, opacity_ipc_handle_);

  // Share tensors over IPC
  share_tensors_over_ipc("127.0.0.1", 65434);
}

GaussianModel::GaussianModel(const GaussianModelParams& model_params)
    : active_sh_degree_(0),
      spatial_lr_scale_(0.0),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1000000) {
  this->max_sh_degree_ = model_params.sh_degree_;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

  // Allocate fixed-size cudaMalloc tensors for IPC
  size_t max_num_gaussians = MAX_NUM_POINTS;

  size_t xyz_num_elements = max_num_gaussians * 3;
  size_t features_dc_num_elements = max_num_gaussians * 1 * 3;
  size_t scaling_num_elements = max_num_gaussians * 3;
  size_t rotation_num_elements = max_num_gaussians * 4;
  size_t opacity_num_elements = max_num_gaussians * 1;

  // Allocate device memory using cudaMalloc
  cudaMalloc(&xyz_ipc_, xyz_num_elements * sizeof(float));
  cudaMalloc(&features_dc_ipc_, features_dc_num_elements * sizeof(float));
  cudaMalloc(&scaling_ipc_, scaling_num_elements * sizeof(float));
  cudaMalloc(&rotation_ipc_, rotation_num_elements * sizeof(float));
  cudaMalloc(&opacity_ipc_, opacity_num_elements * sizeof(float));

  // Initialize
  cudaMemset(opacity_ipc_, 0, opacity_num_elements * sizeof(float));

  // Get IPC handles
  cudaIpcGetMemHandle(&xyz_ipc_handle_, xyz_ipc_);
  cudaIpcGetMemHandle(&features_dc_ipc_handle_, features_dc_ipc_);
  cudaIpcGetMemHandle(&scaling_ipc_handle_, scaling_ipc_);
  cudaIpcGetMemHandle(&rotation_ipc_handle_, rotation_ipc_);
  cudaIpcGetMemHandle(&opacity_ipc_handle_, opacity_ipc_);

  // Collect tensors to share
  tensors_to_share_["xyz"] = std::make_pair(xyz_ipc_, xyz_ipc_handle_);
  tensors_to_share_["features_dc"] = std::make_pair(features_dc_ipc_, features_dc_ipc_handle_);
  tensors_to_share_["scaling"] = std::make_pair(scaling_ipc_, scaling_ipc_handle_);
  tensors_to_share_["rotation"] = std::make_pair(rotation_ipc_, rotation_ipc_handle_);
  tensors_to_share_["opacity"] = std::make_pair(opacity_ipc_, opacity_ipc_handle_);

  // Share tensors over IPC
  share_tensors_over_ipc("127.0.0.1", 65434);
}

void GaussianModel::updateIpcTensors() {
  // Copy data from PyTorch tensors to cudaMalloc tensors
  size_t max_num_gaussians = MAX_NUM_POINTS;

  // For each tensor, copy data up to the current size of the PyTorch tensor or the maximum size
  if (xyz_.numel() > 0) {
    size_t num_elements = xyz_.numel();
    cudaMemcpy(xyz_ipc_, xyz_.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  if (features_dc_.numel() > 0) {
    size_t num_elements = features_dc_.numel();
    cudaMemcpy(features_dc_ipc_, features_dc_.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  if (scaling_.numel() > 0) {
    size_t num_elements = scaling_.numel();
    cudaMemcpy(scaling_ipc_, scaling_.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  if (rotation_.numel() > 0) {
    size_t num_elements = rotation_.numel();
    cudaMemcpy(rotation_ipc_, rotation_.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  if (opacity_.numel() > 0) {
    size_t num_elements = opacity_.numel();
    cudaMemcpy(opacity_ipc_, opacity_.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);

    if (num_elements < MAX_NUM_POINTS) {
      size_t remaining_elements = MAX_NUM_POINTS - num_elements;
      cudaMemset(opacity_ipc_ + num_elements, 0, remaining_elements * sizeof(float));
    }
  }
}

std::string GaussianModel::ipcHandleToHex(const cudaIpcMemHandle_t& ipc_handle) {
  const unsigned char* handle_bytes = reinterpret_cast<const unsigned char*>(&ipc_handle);
  std::ostringstream oss;
  for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(handle_bytes[i]);
  }
  return oss.str();
}

void GaussianModel::send_metadata(const nlohmann::json& metadata_list, const std::string& host, int port) {
  // Set up socket, bind, listen, accept connection
  // Send metadata_length (4-byte big-endian integer)
  // Send metadata_json

  int server_fd, new_socket;
  struct sockaddr_in address;
  int opt = 1;
  socklen_t addrlen = sizeof(address);

  // Create socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
      std::cerr << "Socket failed" << std::endl;
      exit(EXIT_FAILURE);
  }

  // Forcefully attaching socket to the port
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
      std::cerr << "setsockopt failed" << std::endl;
      exit(EXIT_FAILURE);
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = inet_addr(host.c_str());
  address.sin_port = htons(port);

  // Bind
  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
      std::cerr << "Bind failed" << std::endl;
      exit(EXIT_FAILURE);
  }

  // Listen
  if (listen(server_fd, 1) < 0) {
      std::cerr << "Listen failed" << std::endl;
      exit(EXIT_FAILURE);
  }

  std::cout << "Server listening on " << host << ":" << port << "... Waiting for connection." << std::endl;

  // Accept
  if ((new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen)) < 0) {
      std::cerr << "Accept failed" << std::endl;
      exit(EXIT_FAILURE);
  }

  std::cout << "Connected." << std::endl;

  // Serialize metadata to JSON
  std::string metadata_json = metadata_list.dump();

  // Convert to bytes
  const char* metadata_bytes = metadata_json.c_str();
  uint32_t metadata_length = htonl(static_cast<uint32_t>(metadata_json.length()));

  // Send the length
  send(new_socket, &metadata_length, sizeof(metadata_length), 0);

  // Send the metadata
  size_t total_sent = 0;
  size_t to_send = metadata_json.length();

  while (total_sent < to_send) {
      ssize_t sent = send(new_socket, metadata_bytes + total_sent, to_send - total_sent, 0);
      if (sent == -1) {
          std::cerr << "Send failed" << std::endl;
          close(new_socket);
          close(server_fd);
          exit(EXIT_FAILURE);
      }
      total_sent += sent;
  }

  std::cout << "Metadata sent successfully." << std::endl;

  // Optionally, print the sent metadata
  std::cout << "\nSent Metadata:" << std::endl;
  for (size_t idx = 0; idx < metadata_list.size(); ++idx) {
      std::cout << "Tensor " << idx + 1 << " (" << metadata_list[idx]["name"] << "):" << std::endl;
      std::cout << metadata_list[idx].dump(4) << std::endl;
  }

  // Close sockets
  close(new_socket);
  close(server_fd);
}

void GaussianModel::share_tensors_over_ipc(const std::string& host, int port) {
  // Collect metadata
  nlohmann::json metadata_list = nlohmann::json::array();

  for (const auto& pair : tensors_to_share_) {
    const std::string& name = pair.first;
    float* device_ptr = pair.second.first;
    cudaIpcMemHandle_t ipc_handle = pair.second.second;

    nlohmann::json metadata;
    metadata["name"] = name;
    metadata["device"] = "cuda";

    // Serialize ipc_handle to hex
    std::string handle_hex = ipcHandleToHex(ipc_handle);
    metadata["handle"] = handle_hex;

    // For size, we need to know the number of elements
    size_t num_elements = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    if (name == "xyz") {
      shape = { MAX_NUM_POINTS, 3 };
      stride = { 3, 1 };
      num_elements = MAX_NUM_POINTS * 3;
    } else if (name == "features_dc") {
      shape = { MAX_NUM_POINTS, 1, 3 };
      stride = { 3, 3, 1 };
      num_elements = MAX_NUM_POINTS * 1 * 3;
    } else if (name == "scaling") {
      shape = { MAX_NUM_POINTS, 3 };
      stride = { 3, 1 };
      num_elements = MAX_NUM_POINTS * 3;
    } else if (name == "rotation") {
      shape = { MAX_NUM_POINTS, 4 };
      stride = { 4, 1 };
      num_elements = MAX_NUM_POINTS * 4;
    } else if (name == "opacity") {
      shape = { MAX_NUM_POINTS, 1 };
      stride = { 1, 1 };
      num_elements = MAX_NUM_POINTS * 1;
    } else {
      // Handle other tensors if any
    }

    size_t size_in_bytes = num_elements * sizeof(float);
    metadata["size"] = size_in_bytes;
    metadata["offset"] = 0;
    metadata["dtype"] = "torch.float32";
    metadata["storage_size"] = shape;
    metadata["storage_stride"] = stride;
    metadata["storage_offset"] = 0;

    metadata_list.push_back(metadata);
  }

  // Send metadata over TCP
  send_metadata(metadata_list, host, port);
}

torch::Tensor GaussianModel::getScalingActivation() {
  return torch::exp(this->scaling_);
}

torch::Tensor GaussianModel::getRotationActivation() {
  return torch::nn::functional::normalize(this->rotation_);
}

torch::Tensor GaussianModel::getXYZ() { return this->xyz_; }

torch::Tensor GaussianModel::getFeatures() {
  return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()},
                    /*dim=*/1);
}

torch::Tensor GaussianModel::getOpacityActivation() {
  return torch::sigmoid(this->opacity_);
}

torch::Tensor GaussianModel::getCovarianceActivation(int scaling_modifier) {
  // build_rotation
  auto r = this->rotation_;
  auto R = general_utils::build_rotation(r);

  // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
  auto s = scaling_modifier * this->getScalingActivation();
  auto L = torch::zeros(
      {s.size(0), 3, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
  L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
  L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
  L = R.matmul(L);  // L = R @ L

  // build_covariance_from_scaling_rotation
  auto actual_covariance = L.matmul(L.transpose(1, 2));
  return actual_covariance;
}

void GaussianModel::oneUpShDegree() {
  if (this->active_sh_degree_ < this->max_sh_degree_)
    this->active_sh_degree_ += 1;
}

void GaussianModel::setShDegree(const int sh) {
  this->active_sh_degree_ =
      (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void GaussianModel::createFromPcd(std::map<point3D_id_t, Point3D> pcd,
                                  const float spatial_lr_scale) {
  this->spatial_lr_scale_ = spatial_lr_scale;
  int num_points = static_cast<int>(pcd.size());
  torch::Tensor fused_point_cloud = torch::zeros(
      {num_points, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  torch::Tensor color = torch::zeros(
      {num_points, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  auto pcd_it = pcd.begin();
  for (int point_idx = 0; point_idx < num_points; ++point_idx) {
    auto& point = (*pcd_it).second;
    fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
    fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
    fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
    color.index({point_idx, 0}) = point.color_(0);
    color.index({point_idx, 1}) = point.color_(1);
    color.index({point_idx, 2}) = point.color_(2);
    ++pcd_it;
  }

  sparse_points_xyz_ = fused_point_cloud;
  sparse_points_color_ = color;

  torch::Tensor fused_color = sh_utils::RGB2SH(color);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {fused_color.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      fused_color;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points at initialization : " <<
  // fused_point_cloud.size(0) << std::endl;

  torch::Tensor point_cloud_copy = fused_point_cloud.clone();
  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
  torch::Tensor scales = torch::log(torch::sqrt(dist2) * 0.1);
  auto scales_ndimension = scales.ndimension();
  scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  torch::Tensor rots =
      torch::zeros({fused_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor opacities = general_utils::inverse_sigmoid(
      0.5f *
      torch::ones(
          {fused_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  this->exist_since_iter_ = torch::zeros(
      {fused_point_cloud.size(0)},
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  this->xyz_ = fused_point_cloud.requires_grad_();
  this->features_dc_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->features_rest_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->scaling_ = scales.requires_grad_();
  this->rotation_ = rots.requires_grad_();
  this->opacity_ = opacities.requires_grad_();

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::increasePcd(std::vector<float> points,
                                std::vector<float> colors,
                                const int iteration) {
  // auto time1 = std::chrono::steady_clock::now();
  assert(points.size() == colors.size());
  assert(points.size() % 3 == 0);
  auto num_new_points = static_cast<int>(points.size() / 3);
  if (num_new_points == 0) return;

  torch::Tensor new_point_cloud =
      torch::from_blob(points.data(), {num_new_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);
  // torch::zeros({num_new_points, 3}, xyz_.options());
  torch::Tensor new_colors =
      torch::from_blob(colors.data(), {num_new_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);
  // torch::zeros({num_new_points, 3}, xyz_.options());

  sparse_points_xyz_ =
      torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
  sparse_points_color_ =
      torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points increase : "
  //           << num_new_points << std::endl;

  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(new_point_cloud.clone()), 0.0000001);
  torch::Tensor scales = torch::log(torch::sqrt(dist2) * 0.1);
  auto scales_ndimension = scales.ndimension();
  scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  torch::Tensor rots =
      torch::zeros({new_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;
  torch::Tensor opacities = general_utils::inverse_sigmoid(
      0.5f *
      torch::ones(
          {new_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  torch::Tensor new_exist_since_iter = torch::full(
      {new_point_cloud.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_point_cloud;
  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities = opacities;
  auto new_scaling = scales;
  auto new_rotation = rots;

  // auto time2 = std::chrono::steady_clock::now();
  // auto time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
  // std::cout << "increasePcd(umap) preparation time: " << time << " ms"
  // <<std::endl;

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities, new_scaling, new_rotation,
                       new_exist_since_iter);

  c10::cuda::CUDACachingAllocator::emptyCache();
  // auto time3 = std::chrono::steady_clock::now();
  // time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
  // std::cout << "increasePcd(umap) postfix time: " << time << " ms"
  // <<std::endl;
}

void GaussianModel::increasePcd(torch::Tensor& new_point_cloud,
                                torch::Tensor& new_colors,
                                const int iteration) {
  // auto time1 = std::chrono::steady_clock::now();
  auto num_new_points = new_point_cloud.size(0);
  if (num_new_points == 0) return;

  sparse_points_xyz_ =
      torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
  sparse_points_color_ =
      torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points increase : "
  //           << num_new_points << std::endl;

  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(new_point_cloud.clone()), 0.0000001);
  torch::Tensor scales = torch::log(torch::sqrt(dist2) * 0.1);
  auto scales_ndimension = scales.ndimension();
  scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  torch::Tensor rots =
      torch::zeros({new_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;
  torch::Tensor opacities = general_utils::inverse_sigmoid(
      0.5f *
      torch::ones(
          {new_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  torch::Tensor new_exist_since_iter = torch::full(
      {new_point_cloud.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_point_cloud;
  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities = opacities;
  auto new_scaling = scales;
  auto new_rotation = rots;

  // auto time2 = std::chrono::steady_clock::now();
  // auto time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
  // std::cout << "increasePcd(tensor) preparation time: " << time << " ms"
  // <<std::endl;

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities, new_scaling, new_rotation,
                       new_exist_since_iter);

  c10::cuda::CUDACachingAllocator::emptyCache();

  // auto time3 = std::chrono::steady_clock::now();
  // time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
  // std::cout << "increasePcd(tensor) postfix time: " << time << " ms"
  // <<std::endl;
}

void GaussianModel::applyScaledTransformation(const float s,
                                              const Sophus::SE3f T) {
  torch::NoGradGuard no_grad;
  // pt <- (s * Ryw * pt + tyw)
  this->xyz_ *= s;
  torch::Tensor T_tensor =
      tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(this->xyz_, T_tensor);

  // torch::Tensor scales;
  // torch::Tensor point_cloud_copy = this->xyz_.clone();
  // torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy),
  // 0.0000001); scales = torch::log(torch::sqrt(dist2)); auto scales_ndimension
  // = scales.ndimension(); scales =
  // scales.unsqueeze(scales_ndimension).repeat({1, 3});
  this->scaling_ *= s;
  scaledTransformationPostfix(this->xyz_, this->scaling_);
}

void GaussianModel::scaledTransformationPostfix(torch::Tensor& new_xyz,
                                                torch::Tensor& new_scaling) {
  // param_groups[0] = xyz_
  torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(new_xyz, 0);
  // param_groups[4] = scaling_
  torch::Tensor optimizable_scaling =
      this->replaceTensorToOptimizer(new_scaling, 4);

  this->xyz_ = optimizable_xyz;
  this->scaling_ = optimizable_scaling;

  this->Tensor_vec_xyz_ = {this->xyz_};
  this->Tensor_vec_scaling_ = {this->scaling_};
}

void GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale) {
  torch::NoGradGuard no_grad;

  torch::Tensor points = this->getXYZ();
  torch::Tensor rots = this->getRotationActivation();
  // torch::Tensor scales = this->scaling_;// * scale;

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  scaleAndTransformThenMarkVisiblePoints(
      points, rots, point_not_transformed_flags, point_unstable_flags,
      diff_pose, kf_world_view_transform, kf_full_proj_transform,
      num_transformed, scale);

  // torch::Tensor point_cloud_copy = points.clone();
  // torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy),
  // 0.0000001); torch::Tensor scales = torch::log(torch::sqrt(dist2)); auto
  // scales_ndimension = scales.ndimension(); scales =
  // scales.unsqueeze(scales_ndimension).repeat({1, 3});

  // Postfix
  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
  // torch::Tensor optimizable_scaling = this->replaceTensorToOptimizer(scales,
  // 4);
  torch::Tensor optimizable_rots = this->replaceTensorToOptimizer(rots, 5);

  this->xyz_ = optimizable_xyz;
  // this->scaling_ = optimizable_scaling;
  this->rotation_ = optimizable_rots;

  this->Tensor_vec_xyz_ = {this->xyz_};
  // this->Tensor_vec_scaling_ = {this->scaling_};
  this->Tensor_vec_rotation_ = {this->rotation_};
}

void GaussianModel::trainingSetup(
    const GaussianOptimizationParams& training_args) {
  setPercentDense(training_args.percent_dense_);
  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  // this->optimizer_.reset(new torch::optim::Adam(Tensor_vec_xyz_,
  // adam_options));
  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(
      training_args.position_lr_init_ * this->spatial_lr_scale_);

  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);

  // get_expon_lr_func
  lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
  lr_final_ = training_args.position_lr_final_ * this->spatial_lr_scale_;
  lr_delay_mult_ = training_args.position_lr_delay_mult_;
  max_steps_ = training_args.position_lr_max_steps_;
}

float GaussianModel::updateLearningRate(int step) {
  // def update_learning_rate(self, iteration):
  //     ''' Learning rate scheduling per step '''
  //     for param_group in self.optimizer.param_groups:
  //         if param_group["name"] == "xyz":
  //             lr = self.xyz_scheduler_args(iteration)
  //             param_group['lr'] = lr
  //             return lr
  float lr = this->exponLrFunc(step);
  optimizer_->param_groups()[0].options().set_lr(lr);  // Tensor_vec_xyz_
  return lr;
}

// ==================================
// param_groups[0] = xyz_
// param_groups[1] = feature_dc_
// param_groups[2] = feature_rest_
// param_groups[3] = opacity_
// param_groups[4] = scaling_
// param_groups[5] = rotation_
// ==================================
void GaussianModel::setPositionLearningRate(float position_lr) {
  optimizer_->param_groups()[0].options().set_lr(position_lr *
                                                 this->spatial_lr_scale_);
}
void GaussianModel::setFeatureLearningRate(float feature_lr) {
  optimizer_->param_groups()[1].options().set_lr(feature_lr);
  optimizer_->param_groups()[2].options().set_lr(feature_lr / 20.0);
}
void GaussianModel::setOpacityLearningRate(float opacity_lr) {
  optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModel::setScalingLearningRate(float scaling_lr) {
  optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModel::setRotationLearningRate(float rot_lr) {
  optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModel::resetOpacity() {
  torch::Tensor opacities_new = general_utils::inverse_sigmoid(
      torch::min(this->getOpacityActivation(),
                 torch::ones_like(this->getOpacityActivation() * 0.01)));
  torch::Tensor optimizable_tensors =
      this->replaceTensorToOptimizer(opacities_new, 3);  // "opacity"
  this->opacity_ = optimizable_tensors;
  this->Tensor_vec_opacity_ = {this->opacity_};
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor,
                                                      int tensor_idx) {
  auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
  auto& state = optimizer_->state();
  auto key = param.unsafeGetTensorImpl();
  auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
  auto new_state = std::make_unique<torch::optim::AdamParamState>();
  new_state->step(stored_state.step());
  new_state->exp_avg(torch::zeros_like(tensor));
  new_state->exp_avg_sq(torch::zeros_like(tensor));
  // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed
  // only when options.amsgrad(true), which is false by default

  state.erase(key);
  param = tensor.requires_grad_();
  key = param.unsafeGetTensorImpl();
  state[key] = std::move(new_state);

  auto optimizable_tensors = param;
  return optimizable_tensors;
}

void GaussianModel::prunePoints(torch::Tensor& mask) {
  auto valid_points_mask = ~mask;

  // _prune_optimizer
  std::vector<torch::Tensor> optimizable_tensors(6);
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& param = param_groups[group_idx].params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(
          stored_state.exp_avg().index({valid_points_mask}).clone());
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index({valid_points_mask}).clone());
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = param.index({valid_points_mask}).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
      optimizable_tensors[group_idx] = param;
    } else {
      param = param.index({valid_points_mask}).requires_grad_();
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ = this->exist_since_iter_.index({valid_points_mask});

  this->xyz_gradient_accum_ =
      this->xyz_gradient_accum_.index({valid_points_mask});

  this->denom_ = this->denom_.index({valid_points_mask});
  this->max_radii2D_ = this->max_radii2D_.index({valid_points_mask});
}

void GaussianModel::densificationPostfix(torch::Tensor& new_xyz,
                                         torch::Tensor& new_features_dc,
                                         torch::Tensor& new_features_rest,
                                         torch::Tensor& new_opacities,
                                         torch::Tensor& new_scaling,
                                         torch::Tensor& new_rotation,
                                         torch::Tensor& new_exist_since_iter) {
  // cat_tensors_to_optimizer
  std::vector<torch::Tensor> optimizable_tensors(6);
  std::vector<torch::Tensor> tensors_dict = {new_xyz,           new_features_dc,
                                             new_features_rest, new_opacities,
                                             new_scaling,       new_rotation};
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& group = param_groups[group_idx];
    assert(group.params().size() == 1);
    auto& extension_tensor = tensors_dict[group_idx];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(torch::cat(
          {stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)},
          /*dim=*/0));
      new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(),
                                        torch::zeros_like(extension_tensor)},
                                       /*dim=*/0));
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);

      optimizable_tensors[group_idx] = param;
    } else {
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ =
      torch::cat({this->exist_since_iter_, new_exist_since_iter}, /*dim=*/0);

  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));
  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::densifyAndSplit(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent,
                                    int N) {
  int n_init_points = this->getXYZ().size(0);
  // Extract points that satisfy the gradient condition
  auto padded_grad = torch::zeros({n_init_points},
                                  torch::TensorOptions().device(device_type_));
  padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0))
      .copy_(grads.squeeze());
  auto selected_pts_mask =
      torch::where(padded_grad >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) >
          percentDense() * scene_extent);

  auto new_xyz = this->getXYZ().index({selected_pts_mask}).repeat({N, 1});
  auto samples = torch::normal(0.0f, 1.0f, new_xyz.sizes()).to(device_type_);
  samples.index_put_({samples.abs() > 3.0f}, 0.0f);
  auto new_covs = this->getCovarianceActivation()
                      .index({selected_pts_mask})
                      .repeat({N, 1, 1});
  new_xyz += new_covs.matmul(samples.unsqueeze(-1)).squeeze(-1);
  auto new_scaling = torch::log(
      this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1}) /
      (0.8 * N));  // scaling_inverse_activation
  auto new_rotation = this->rotation_.index({selected_pts_mask}).repeat({N, 1});
  auto new_features_dc =
      this->features_dc_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_features_rest =
      this->features_rest_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_opacity = this->opacity_.index({selected_pts_mask}).repeat({N, 1});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask}).repeat({N});

  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacity, new_scaling, new_rotation,
                             new_exist_since_iter);

  auto prune_filter = torch::cat(
      {selected_pts_mask,
       torch::zeros(
           {(N * selected_pts_mask.sum()).item<int>()},
           torch::TensorOptions().device(device_type_).dtype(torch::kBool))});
  this->prunePoints(prune_filter);
}

void GaussianModel::densifyAndClone(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent) {
  // Extract points that satisfy the gradient condition
  auto selected_pts_mask = torch::where(
      torch::frobenius_norm(grads, /*dim=*/-1) >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) <=
          percentDense() * scene_extent);

  auto new_xyz = this->xyz_.index({selected_pts_mask});
  auto new_features_dc = this->features_dc_.index({selected_pts_mask});
  auto new_features_rest = this->features_rest_.index({selected_pts_mask});
  auto new_opacities = this->opacity_.index({selected_pts_mask});
  auto new_scaling = this->scaling_.index({selected_pts_mask});
  auto new_rotation = this->rotation_.index({selected_pts_mask});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask});

  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacities, new_scaling, new_rotation,
                             new_exist_since_iter);
}

void GaussianModel::densifyAndPrune(float max_grad,
                                    float min_opacity,
                                    float extent,
                                    int max_screen_size) {
  auto grads = this->xyz_gradient_accum_ / this->denom_;
  grads.index_put_({grads.isnan()}, 0.0f);
  this->densifyAndClone(grads, max_grad, extent);
  this->densifyAndSplit(grads, max_grad, extent);

  auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
  if (max_screen_size) {
    auto big_points_vs = this->max_radii2D_ > max_screen_size;
    auto big_points_ws =
        std::get<0>(this->getScalingActivation().max(/*dim=*/1)) >
        0.1f * extent;
    prune_mask = torch::logical_or(torch::logical_or(prune_mask, big_points_vs),
                                   big_points_ws);
  }
  this->prunePoints(prune_mask);

  c10::cuda::CUDACachingAllocator::emptyCache();  // torch.cuda.empty_cache()
}

void GaussianModel::addDensificationStats(torch::Tensor& viewspace_point_tensor,
                                          torch::Tensor& update_filter) {
  this->xyz_gradient_accum_.index_put_(
      {update_filter},
      torch::frobenius_norm(viewspace_point_tensor.grad().index(
                                {update_filter, torch::indexing::Slice(0, 2)}),
                            /*dim=*/-1,
                            /*keepdim=*/true),
      /*accumulate=*/true);

  this->denom_.index_put_({update_filter},
                          this->denom_.index({update_filter}) + 1);
}

// void GaussianModel::increasePointsIterationsOfExistence(const int i)
// {
//     this->exist_since_iter_ += i;
// }

void GaussianModel::loadPly(std::filesystem::path ply_path) {
  std::ifstream instream_binary(ply_path, std::ios::binary);
  if (!instream_binary.is_open() || instream_binary.fail())
    throw std::runtime_error("Fail to open ply file at " + ply_path.string());
  instream_binary.seekg(0, std::ios::beg);

  tinyply::PlyFile ply_file;
  ply_file.parse_header(instream_binary);

  std::cout << "\t[ply_header] Type: "
            << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
  for (const auto& c : ply_file.get_comments())
    std::cout << "\t[ply_header] Comment: " << c << std::endl;
  for (const auto& c : ply_file.get_info())
    std::cout << "\t[ply_header] Info: " << c << std::endl;

  for (const auto& e : ply_file.get_elements()) {
    std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
              << std::endl;
    for (const auto& p : e.properties) {
      std::cout << "\t[ply_header] \tproperty: " << p.name
                << " (type=" << tinyply::PropertyTable[p.propertyType].str
                << ")";
      if (p.isList)
        std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                  << ")";
      std::cout << std::endl;
    }
  }

  std::shared_ptr<tinyply::PlyData> xyz, f_dc, f_rest, opacity, scales, rot;

  try {
    xyz = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    f_dc = ply_file.request_properties_from_element(
        "vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  int n_f_rest = ((max_sh_degree_ + 1) * (max_sh_degree_ + 1) - 1) * 3;
  if (n_f_rest >= 0) {
    std::vector<std::string> f_rest_element_names(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
      f_rest_element_names[i] = "f_rest_" + std::to_string(i);
    try {
      f_rest = ply_file.request_properties_from_element("vertex",
                                                        f_rest_element_names);
    } catch (const std::exception& e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
  }

  try {
    opacity = ply_file.request_properties_from_element("vertex", {"opacity"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    scales = ply_file.request_properties_from_element(
        "vertex", {"scale_0", "scale_1", "scale_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    rot = ply_file.request_properties_from_element(
        "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  ply_file.read(instream_binary);

  if (xyz) std::cout << "\tRead " << xyz->count << " total xyz " << std::endl;
  if (f_dc)
    std::cout << "\tRead " << f_dc->count << " total f_dc " << std::endl;
  if (f_rest)
    std::cout << "\tRead " << f_rest->count << " total f_rest " << std::endl;
  if (opacity)
    std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
  if (scales)
    std::cout << "\tRead " << scales->count << " total scales " << std::endl;
  if (rot) std::cout << "\tRead " << rot->count << " total rot " << std::endl;

  // Data to std::vector
  const int num_points = xyz->count;

  const std::size_t n_xyz_bytes = xyz->buffer.size_bytes();
  std::vector<float> xyz_vector(xyz->count * 3);
  std::memcpy(xyz_vector.data(), xyz->buffer.get(), n_xyz_bytes);

  const std::size_t n_f_dc_bytes = f_dc->buffer.size_bytes();
  std::vector<float> f_dc_vector(f_dc->count * 3);
  std::memcpy(f_dc_vector.data(), f_dc->buffer.get(), n_f_dc_bytes);

  const std::size_t n_f_rest_bytes = f_rest->buffer.size_bytes();
  std::vector<float> f_rest_vector(f_rest->count * n_f_rest);
  std::memcpy(f_rest_vector.data(), f_rest->buffer.get(), n_f_rest_bytes);

  const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
  std::vector<float> opacity_vector(opacity->count * 1);
  std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

  const std::size_t n_scales_bytes = scales->buffer.size_bytes();
  std::vector<float> scales_vector(scales->count * 3);
  std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

  const std::size_t n_rot_bytes = rot->buffer.size_bytes();
  std::vector<float> rot_vector(rot->count * 4);
  std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

  // std::vector to torch::Tensor
  this->xyz_ = torch::from_blob(xyz_vector.data(), {num_points, 3},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .to(device_type_);

  this->features_dc_ =
      torch::from_blob(f_dc_vector.data(), {num_points, 3, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->features_rest_ =
      torch::from_blob(f_rest_vector.data(), {num_points, 3, n_f_rest / 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->opacity_ =
      torch::from_blob(opacity_vector.data(), {num_points, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->scaling_ =
      torch::from_blob(scales_vector.data(), {num_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->rotation_ =
      torch::from_blob(rot_vector.data(), {num_points, 4},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->active_sh_degree_ = this->max_sh_degree_;
}

void GaussianModel::savePly(std::filesystem::path result_path) {
  // Prepare data to write
  torch::Tensor xyz = this->xyz_.detach().cpu();
  torch::Tensor normals = torch::zeros_like(xyz);
  torch::Tensor f_dc =
      this->features_dc_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
  torch::Tensor f_rest = this->features_rest_.detach()
                             .transpose(1, 2)
                             .flatten(1)
                             .contiguous()
                             .cpu();
  torch::Tensor opacities = this->opacity_.detach().cpu();
  torch::Tensor scale = this->scaling_.detach().cpu();
  torch::Tensor rotation = this->rotation_.detach().cpu();

  std::filebuf fb_binary;
  fb_binary.open(result_path, std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + result_path.string());

  tinyply::PlyFile result_file;

  // xyz
  result_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.size(0),
      reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()), tinyply::Type::INVALID,
      0);

  // normals
  result_file.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normals.size(0),
      reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_dc
  std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
  std::vector<std::string> property_names_f_dc(n_f_dc);
  for (int i = 0; i < n_f_dc; ++i)
    property_names_f_dc[i] = "f_dc_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_dc, tinyply::Type::FLOAT32,
      this->features_dc_.size(0),
      reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_rest
  std::size_t n_f_rest =
      this->features_rest_.size(1) * this->features_rest_.size(2);
  std::vector<std::string> property_names_f_rest(n_f_rest);
  for (int i = 0; i < n_f_rest; ++i)
    property_names_f_rest[i] = "f_rest_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_rest, tinyply::Type::FLOAT32,
      this->features_rest_.size(0),
      reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // opacities
  result_file.add_properties_to_element(
      "vertex", {"opacity"}, tinyply::Type::FLOAT32, opacities.size(0),
      reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // scale
  std::size_t n_scale = scale.size(1);
  std::vector<std::string> property_names_scale(n_scale);
  for (int i = 0; i < n_scale; ++i)
    property_names_scale[i] = "scale_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_scale, tinyply::Type::FLOAT32, scale.size(0),
      reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // rotation
  std::size_t n_rotation = rotation.size(1);
  std::vector<std::string> property_names_rotation(n_rotation);
  for (int i = 0; i < n_rotation; ++i)
    property_names_rotation[i] = "rot_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_rotation, tinyply::Type::FLOAT32,
      rotation.size(0), reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // Write the file
  result_file.write(outstream_binary, true);

  fb_binary.close();
}

void GaussianModel::saveSparsePointsPly(std::filesystem::path result_path) {
  // Prepare data to write
  torch::Tensor xyz = this->sparse_points_xyz_.detach().cpu();
  torch::Tensor normals = torch::zeros_like(xyz);
  torch::Tensor color = (this->sparse_points_color_ * 255.0f)
                            .toType(torch::kUInt8)
                            .detach()
                            .cpu();

  std::filebuf fb_binary;
  fb_binary.open(result_path, std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + result_path.string());

  tinyply::PlyFile result_file;

  // xyz
  result_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.size(0),
      reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()), tinyply::Type::INVALID,
      0);

  // normals
  result_file.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normals.size(0),
      reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // color
  result_file.add_properties_to_element(
      "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, color.size(0),
      reinterpret_cast<uint8_t*>(color.data_ptr<uint8_t>()),
      tinyply::Type::INVALID, 0);

  // Write the file
  result_file.write(outstream_binary, true);

  fb_binary.close();
}

float GaussianModel::percentDense() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return percent_dense_;
}

void GaussianModel::setPercentDense(const float percent_dense) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  percent_dense_ = percent_dense;
}

/**
 * @brief get_expon_lr_func
 * @details Modified from Plenoxels
 *  Continuous learning rate decay function. Adapted from JaxNeRF
 *  The returned rate is lr_init when step=0 and lr_final when step=max_steps,
 * and is log-linearly interpolated elsewhere (equivalent to exponential decay).
 *  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
 *  function of lr_delay_mult, such that the initial learning rate is
 *  lr_init*lr_delay_mult at the beginning of optimization but will be eased
 * back to the normal learning rate when steps>lr_delay_steps. :param conf:
 * config subtree 'lr' or similar :param max_steps: int, the number of steps
 * during optimization. :return HoF which takes step as input
 * @param iteration
 * @return float
 */
float GaussianModel::exponLrFunc(int step) {
  if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f)) return 0.0f;

  float delay_rate;
  if (lr_delay_steps_ > 0)
    delay_rate = lr_delay_mult_ +
                 (1.0f - lr_delay_mult_) *
                     std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) /
                                                         lr_delay_steps_,
                                                     0.0f, 1.0f));
  else
    delay_rate = 1.0f;
  float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
  float log_lerp =
      std::exp(std::log(lr_init_) * (1 - t) + std::log(lr_final_) * t);
  return delay_rate * log_lerp;
}
