#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <rasterizer.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cfloat>

#include <nlohmann/json.hpp> // For JSON parsing
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring> // For memcpy
#include <vector>
#include <unordered_map>

using json = nlohmann::json;
using namespace std;

typedef	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> Vector3i;

inline std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

template<typename T> float* remove_cuda(float* cuda, size_t sz, size_t pos, size_t nb) {
	return 0;
}

//Gaussian Splatting data structure
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

template<int D> int loadPly(const char* filename, std::vector<Pos>& pos, std::vector<SHs<3>>& shs, std::vector<float>& opacities, std::vector<Scale>& scales, std::vector<Rot>& rot, Vector3f& minn, Vector3f& maxx);

void GaussianSplattingRenderer::SetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			it->_boxmin = Vector3f(box_min);
			it->_boxmax = Vector3f(box_max);
			break;
		}
	}
}

void GaussianSplattingRenderer::GetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			box_min[0] = it->_scenemin.x();
			box_min[1] = it->_scenemin.y();
			box_min[2] = it->_scenemin.z();
			box_max[0] = it->_scenemax.x();
			box_max[1] = it->_scenemax.y();
			box_max[2] = it->_scenemax.z();
			break;
		}
	}
}

int GaussianSplattingRenderer::GetNbSplat() {
	return count;
}

void GaussianSplattingRenderer::Load(const char* file) {
	count_cpu = 0;
}

bool receive_metadata_over_tcp(json& metadata_list, const std::string& host = "127.0.0.1", int port = 65434) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error." << std::endl;
        return false;
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IPv4 addresses from text to binary form
    if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported." << std::endl;
        return false;
    }

    std::cout << "Connecting to " << host << ":" << port << "..." << std::endl;

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection Failed." << std::endl;
        return false;
    }

    // Receive the length of the metadata first
    uint32_t metadata_length_net;
    ssize_t valread = read(sock, &metadata_length_net, sizeof(metadata_length_net));
    if (valread != sizeof(metadata_length_net)) {
        std::cerr << "Failed to read metadata length." << std::endl;
        close(sock);
        return false;
    }
    uint32_t metadata_length = ntohl(metadata_length_net);

    // Receive the metadata
    std::vector<char> buffer(metadata_length);
    size_t total_received = 0;
    while (total_received < metadata_length) {
        ssize_t received = read(sock, buffer.data() + total_received, metadata_length - total_received);
        if (received <= 0) {
            std::cerr << "Failed to read metadata." << std::endl;
            close(sock);
            return false;
        }
        total_received += received;
    }

    close(sock);

    std::string metadata_str(buffer.begin(), buffer.end());
    metadata_list = json::parse(metadata_str);

    std::cout << "Metadata received." << std::endl;

    return true;
}

int GaussianSplattingRenderer::CopyToCuda() {
    const std::lock_guard<std::mutex> lock(cuda_mtx);

    // Receive metadata over TCP
    json metadata_list;
    if (!receive_metadata_over_tcp(metadata_list)) {
        // Handle error
        return -1;
    }

	int64_t num_gaussians = 0;

    // Vector to hold the tensors' information
    struct TensorInfo {
        void* tensor_ptr;
        std::vector<int64_t> shape;
        std::vector<int64_t> stride;
        std::string dtype_str;
        std::string tensor_name; // Added tensor_name
    };
    std::vector<TensorInfo> tensors;

    // Map to store storage handle to device pointer
    std::unordered_map<std::string, void*> handle_to_dev_ptr;

    // Loop over each metadata entry to reconstruct tensors
    for (size_t idx = 0; idx < metadata_list.size(); ++idx) {
        const auto& metadata = metadata_list[idx];

        // Print metadata for each tensor
        std::cout << "\nProcessing Tensor " << idx << " metadata:" << std::endl;
        std::cout << metadata.dump(4) << std::endl;

        // Get the IPC handle from the metadata
        std::string handle_hex = metadata["handle"];

        // Check if we've already opened this handle
        void* dev_ptr = nullptr;
        if (handle_to_dev_ptr.find(handle_hex) != handle_to_dev_ptr.end()) {
            // Handle already opened
            dev_ptr = handle_to_dev_ptr[handle_hex];
        } else {
            // Convert handle_hex to cudaIpcMemHandle_t
            std::vector<unsigned char> handle_bytes;
            for (size_t i = 0; i < handle_hex.length(); i += 2) {
                std::string byteString = handle_hex.substr(i, 2);
                unsigned char byte = static_cast<unsigned char>(strtol(byteString.c_str(), nullptr, 16));
                handle_bytes.push_back(byte);
            }

            cudaIpcMemHandle_t ipc_handle;
            if (handle_bytes.size() != sizeof(cudaIpcMemHandle_t)) {
                std::cerr << "Invalid handle size for tensor " << idx << "." << std::endl;
                return -1;
            }
            memcpy(&ipc_handle, handle_bytes.data(), sizeof(cudaIpcMemHandle_t));

            // Open the IPC memory handle to get the device pointer
            cudaError_t err = cudaIpcOpenMemHandle(&dev_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                std::cerr << "cudaIpcOpenMemHandle failed for tensor " << idx << ": " << cudaGetErrorString(err) << std::endl;
                return -1;
            }
            // Store the device pointer for this handle
            handle_to_dev_ptr[handle_hex] = dev_ptr;
        }

        // Retrieve tensor properties
        int64_t storage_offset_bytes = metadata["offset"];
        std::vector<int64_t> size = metadata["storage_size"].get<std::vector<int64_t>>();
        std::vector<int64_t> stride = metadata["storage_stride"].get<std::vector<int64_t>>();

        // Get the data type
        std::string dtype_str = metadata["dtype"];

        // Get the tensor name (assuming it's provided in the metadata)
        std::string tensor_name = metadata["name"];

        // Adjust the device pointer by storage offset in bytes
        float* tensor_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(dev_ptr) + storage_offset_bytes);

        // Now assign the tensor_ptr to the appropriate variable
        if (tensor_name == "xyz") {
            pos_cuda = tensor_ptr;
            num_gaussians = size[0]; // Number of Gaussians
        } else if (tensor_name == "features_dc") {
            shs_cuda = tensor_ptr;
        } else if (tensor_name == "scaling") {
            scale_cuda = tensor_ptr;
        } else if (tensor_name == "rotation") {
            rot_cuda = tensor_ptr;
        } else if (tensor_name == "opacity") {
            opacity_cuda = tensor_ptr;
        } else {
            // Unknown tensor
            std::cerr << "Unknown tensor name: " << tensor_name << std::endl;
        }
    }

    count += static_cast<int>(num_gaussians);
	// count = 50000;

	// Register new model
    model_idx += 1;
	
    models.push_back({ model_idx, count, false, _scenemin, _scenemax, _scenemin, _scenemax });

    // Allocate or reallocate working buffers
    if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
    if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

    bool white_bg = false;
    float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    AllocateRenderContexts();

    // Return the new model index
    return model_idx;
}

void GaussianSplattingRenderer::RemoveModel(int model) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	size_t start = 0;
	std::list<SplatModel>::iterator mit = models.end();
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			mit = it;
			break;
		}
		start += it->size;
	}

	if (mit != models.end()) {
		size_t size = mit->size;
		pos_cuda = remove_cuda<Pos>(pos_cuda, count, start, size);
		rot_cuda = remove_cuda<Rot>(rot_cuda, count, start, size);
		shs_cuda = remove_cuda<SHs<0>>(shs_cuda, count, start, size);
		opacity_cuda = remove_cuda<float>(opacity_cuda, count, start, size);
		scale_cuda = remove_cuda<Scale>(scale_cuda, count, start, size);

		count -= size;
		models.erase(mit);

		//Working buffer or fixed data
		//can be fully reallocated
		if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
		if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

		bool white_bg = false;
		float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

		AllocateRenderContexts();

	} else {
		throw std::runtime_error("Model index not found.");
	}
}

void GaussianSplattingRenderer::CreateRenderContext(int idx) {

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	//Resize the buffers
	geom[idx] = new AllocFuncBuffer;
	binning[idx] = new AllocFuncBuffer;
	img[idx] = new AllocFuncBuffer;
	renData[idx] = new RenderData;

	//Alloc
	geom[idx]->bufferFunc = resizeFunctional(&geom[idx]->ptr, geom[idx]->allocd);
	binning[idx]->bufferFunc = resizeFunctional(&binning[idx]->ptr, binning[idx]->allocd);
	img[idx]->bufferFunc = resizeFunctional(&img[idx]->ptr, img[idx]->allocd);

	//Alloc cuda ressource for view model
	AllocateRenderContexts();
}

void GaussianSplattingRenderer::RemoveRenderContext(int idx) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	//freee cuda resources
	if (geom.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)geom.at(idx)->ptr)); }
	if (binning.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)binning.at(idx)->ptr)); }
	if (img.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)img.at(idx)->ptr)); }

	geom.erase(idx);
	binning.erase(idx);
	img.erase(idx);

	if (renData.at(idx)->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->view_cuda)); }
	if (renData.at(idx)->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->proj_cuda)); }
	if (renData.at(idx)->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_sz)); }
	if (renData.at(idx)->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_active)); }
	if (renData.at(idx)->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->cam_pos_cuda)); }
	if (renData.at(idx)->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmin)); }
	if (renData.at(idx)->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmax)); }
	if (renData.at(idx)->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->frustums)); }

	RenderData* data = renData.at(idx);
	renData.erase(idx);
	delete data;
}

void GaussianSplattingRenderer::AllocateRenderContexts() {
	size_t nb_models = models.size();
	for (auto kv: renData) {
		RenderData* data = kv.second;
		//reallocate only if needed
		if (data->nb_model_allocated != nb_models) {
			data->nb_model_allocated = nb_models;
			
			//free last allocated ressources
			if (data->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->view_cuda))); }
			if (data->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->proj_cuda))); }
			if (data->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_sz))); }
			if (data->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_active))); }
			if (data->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->cam_pos_cuda))); }
			if (data->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmin))); }
			if (data->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmax))); }
			if (data->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->frustums))); }

			// Create space for view parameters for each model
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->view_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->proj_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_sz), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_active), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->cam_pos_cuda), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmin), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmax), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->frustums), 6 * sizeof(float)));
		}
	}
}

void GaussianSplattingRenderer::SetActiveModel(int model, bool active) {
	for (SplatModel& m : models) {
		if (m.index == model) {
			m.active = active;
		}
	}
}

void GaussianSplattingRenderer::Preprocess(int context, const std::map<int, Matrix4f>& view_mat, const std::map<int, Matrix4f>& proj_mat, const std::map<int, Vector3f>& position, Vector6f frumstums, float fovy, int width, int height) {
	//view_mat.row(1) *= -1;
	//view_mat.row(2) *= -1;
	//proj_mat.row(1) *= -1;

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	if (count == 0) { return; }
	
	float aspect_ratio = (float)width / (float)height;
	float tan_fovy = tan(fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	RenderData* rdata = renData.at(context);
	int nb_models = models.size();
	int midx = 0;
	for (const SplatModel& m : models) {
		int active = (m.active && view_mat.find(m.index) != view_mat.end()) ? 1 : 0;
		int msize = m.size;
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_sz) + midx * sizeof(int), &msize, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_active) + midx * sizeof(int), &active, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmin) + midx * sizeof(float) * 3, m._boxmin.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmax) + midx * sizeof(float) * 3, m._boxmax.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		if (active == 1) {
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->view_cuda) + midx * sizeof(Matrix4f), view_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->proj_cuda) + midx * sizeof(Matrix4f), proj_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->cam_pos_cuda) + midx * sizeof(float) * 3, position.at(m.index).data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		}
		midx += 1;
	}
	CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->frustums), frumstums.data(), sizeof(float) * 6, cudaMemcpyHostToDevice));

	// Rasterize
	int* rects = _fastCulling ? rect_cuda : nullptr;
	rdata->num_rendered = CudaRasterizer::Rasterizer::forward_preprocess(
		geom.at(context)->bufferFunc,
		binning.at(context)->bufferFunc,
		img.at(context)->bufferFunc,
		count, 0, 1,
		background_cuda,
		width, height,
		pos_cuda,
		shs_cuda,
		nullptr,
		opacity_cuda,
		scale_cuda,
		_scalingModifier,
		rot_cuda,
		nullptr,
		rdata->view_cuda,
		rdata->proj_cuda,
		rdata->cam_pos_cuda,
		rdata->frustums,
		rdata->model_sz,
		rdata->model_active,
		nb_models,
		tan_fovx,
		tan_fovy,
		false,
		nullptr,
		rects,
		rdata->boxmin,
		rdata->boxmax);
}

void GaussianSplattingRenderer::Render(int context, float* image_cuda, float* depth_cuda, cudaSurfaceObject_t camera_depth_cuda, float fovy, int width, int height) {
	if (count > 0 && renData.at(context)->num_rendered > 0) {
		
		RenderData* rdata = renData.at(context);
		
		const std::lock_guard<std::mutex> lock(cuda_mtx);
		
		float aspect_ratio = (float)width / (float)height;
		float tan_fovy = tan(fovy * 0.5f);
		float tan_fovx = tan_fovy * aspect_ratio;

		int* rects = _fastCulling ? rect_cuda : nullptr;

		CudaRasterizer::Rasterizer::forward_render(
			geom.at(context)->bufferFunc,
			binning.at(context)->bufferFunc,
			img.at(context)->bufferFunc,
			count, 0, 1,
			background_cuda,
			camera_depth_cuda,
			width, height,
			pos_cuda,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			rot_cuda,
			nullptr,
			rdata->view_cuda,
			rdata->proj_cuda,
			rdata->cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			depth_cuda,
			nullptr,
			rects,
			rdata->boxmin,
			rdata->boxmax,
			rdata->num_rendered);
	} else {
		CUDA_SAFE_CALL(cudaMemset(image_cuda, 0, sizeof(float) * 4 * width * height));
		CUDA_SAFE_CALL(cudaMemset(depth_cuda, 0, sizeof(float) * width * height));
	}
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	Vector3f& minn,
	Vector3f& maxx)
{
    return 0;
}