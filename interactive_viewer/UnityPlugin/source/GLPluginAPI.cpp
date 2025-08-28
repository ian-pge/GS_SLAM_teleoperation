#include "GLPluginAPI.h"
#include "CudaKernels.h"
#include "PlatformBase.h"

#include <assert.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

#include <sstream>

using namespace std;

inline bool gl_error(const char* func, int line, std::string& _message) {
#if DEBUG || _DEBUG
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		_message.assign((stringstream()<<func<<"::"<<line<< " OpenglError: 0x" << hex << err).str());
		return true;
	}
#endif
	return false;
}


GLPluginAPI::GLPluginAPI() {
#if UNITY_WIN && SUPPORT_OPENGL_CORE
	gl3wInit();
#endif
}

GLPluginAPI::GLPOV::~GLPOV() {
	if (imageBuffer) { glDeleteTextures(1, &imageBuffer); }
}

GLPluginAPI::~GLPluginAPI() {
}

bool GLPluginAPI::GLPOV::Init(string& message) {
	//cuda interop
	if (imageBuffer) { glDeleteTextures(1, &imageBuffer); }
	POV::FreeCudaRessources();

	//Alloc a new splat buffer for results
	POV::AllocSplatBuffer(message);

	//Alloc Texture for final result
	glGenTextures(1, &imageBuffer);
	glBindTexture(GL_TEXTURE_2D, imageBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	if (cudaPeekAtLastError() != cudaSuccess) { message = cudaGetErrorString(cudaGetLastError()); return false; }
	cudaGraphicsGLRegisterImage(&imageBufferCuda, imageBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_interop_failed = !(cudaPeekAtLastError() == cudaSuccess);

	if (!_interop_failed) {
		glGenTextures(1, &depthBuffer);
		glBindTexture(GL_TEXTURE_2D, depthBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		if (cudaPeekAtLastError() != cudaSuccess) { message = cudaGetErrorString(cudaGetLastError()); return false; }
		cudaGraphicsGLRegisterImage(&imageDepthBufferCuda, depthBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		_interop_failed = !(cudaPeekAtLastError() == cudaSuccess);
	}

	if (!_interop_failed) {
		cudaGraphicsGLRegisterImage(&imageCameraDepthBufferCuda, cameraDepthBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		_interop_failed = !(cudaPeekAtLastError() == cudaSuccess);
	}

	return POV::AllocFallbackIfNeeded(message);
}

void* GLPluginAPI::GLPOV::GetTextureNativePointer() {
	return reinterpret_cast<void*>(static_cast<uintptr_t>(imageBuffer));
}

void* GLPluginAPI::GLPOV::GetDepthTextureNativePointer() {
	return reinterpret_cast<void*>(static_cast<uintptr_t>(depthBuffer));
}

void GLPluginAPI::GLPOV::SetCameraDepthTextureNativePointer(void* ptr) {
	cameraDepthBuffer = static_cast<GLuint>(reinterpret_cast<uintptr_t>(ptr));
}

POV* GLPluginAPI::CreatePOV() {
	return new GLPOV;
}

void GLPluginAPI::Init()
{
	int num_devices;
	cudaGetDeviceCount(&num_devices); if (CUDA_ERROR(_message)) { return; }

	_device = 0;
	if (_device >= num_devices) {
		_message = "No CUDA devices detected!";
		return;
	}

	if (!PluginAPI::SetAndCheckCudaDevice()) { return; }
	PluginAPI::InitPovs();
}
