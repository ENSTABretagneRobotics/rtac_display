#include <rtac_display/cuda/utils.h>

#include <cuda_runtime.h> // check if needed
#include <cuda_gl_interop.h>

namespace rtac { namespace display { namespace cuda {

void copy_to_gl(GLuint bufferId, const void* cudaDevicePtr, size_t byteCount)
{
    cudaGraphicsResource* cudaResource(NULL);
    void* GLdevicePtr = NULL;
    size_t accessibleSize = 0;

    CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
        &cudaResource, bufferId, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK( cudaGraphicsMapResources(1, &cudaResource));
    
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
        &GLdevicePtr, &accessibleSize, cudaResource));
    
    if(accessibleSize < byteCount) {
        std::ostringstream oss;
        oss << "GL buffer not big enough for copy (needs " << byteCount 
            << "b, buffer is " << accessibleSize << "b)";
        throw std::runtime_error(oss.str());
    }

    CUDA_CHECK(cudaMemcpy(GLdevicePtr, cudaDevicePtr, byteCount,
                              cudaMemcpyDeviceToDevice));

    CUDA_CHECK( cudaGraphicsUnmapResources(1, &cudaResource));
    CUDA_CHECK( cudaGraphicsUnregisterResource(cudaResource));
}

void copy_from_gl(void* cudaDevicePtr, GLuint bufferId, size_t byteCount)
{
    cudaGraphicsResource* cudaResource(NULL);
    void* GLdevicePtr = NULL;
    size_t accessibleSize = 0;

    CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
        &cudaResource, bufferId, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK( cudaGraphicsMapResources(1, &cudaResource));
    
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
        &GLdevicePtr, &accessibleSize, cudaResource));
    
    if(accessibleSize < byteCount) {
        std::ostringstream oss;
        oss << "GL buffer not big enough for copy (needs " << byteCount 
            << "b, buffer is " << accessibleSize << "b)";
        throw std::runtime_error(oss.str());
    }

    CUDA_CHECK(cudaMemcpy(cudaDevicePtr, GLdevicePtr, byteCount,
                              cudaMemcpyDeviceToDevice));

    CUDA_CHECK( cudaGraphicsUnmapResources(1, &cudaResource));
    CUDA_CHECK( cudaGraphicsUnregisterResource(cudaResource));
}

}; //namespace cuda
}; //namespace display
}; //namespace rtac
