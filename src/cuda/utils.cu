#include <rtac_display/cuda/utils.h>

#include <cuda_runtime.h> // check if needed
#include <cuda_gl_interop.h>

namespace rtac { namespace display { namespace cuda {

void copy_to_gl(GLuint bufferId, const void* cudaDevicePtr, size_t byteCount)
{
    cudaGraphicsResource* cudaResource(NULL);
    void* GLdevicePtr = NULL;
    size_t accessibleSize = 0;

    rtac::cuda::check_error( cudaGraphicsGLRegisterBuffer(
        &cudaResource, bufferId, cudaGraphicsRegisterFlagsWriteDiscard));
    rtac::cuda::check_error( cudaGraphicsMapResources(1, &cudaResource));
    
    rtac::cuda::check_error( cudaGraphicsResourceGetMappedPointer(
        &GLdevicePtr, &accessibleSize, cudaResource));
    
    if(accessibleSize < byteCount) {
        std::ostringstream oss;
        oss << "GL buffer not big enough for copy (needs " << byteCount 
            << "b, buffer is " << accessibleSize << "b)";
        throw std::runtime_error(oss.str());
    }

    rtac::cuda::check_error(cudaMemcpy(GLdevicePtr, cudaDevicePtr, byteCount,
                              cudaMemcpyDeviceToDevice));

    rtac::cuda::check_error( cudaGraphicsUnmapResources(1, &cudaResource));
    rtac::cuda::check_error( cudaGraphicsUnregisterResource(cudaResource));
}

void copy_from_gl(void* cudaDevicePtr, GLuint bufferId, size_t byteCount)
{
    cudaGraphicsResource* cudaResource(NULL);
    void* GLdevicePtr = NULL;
    size_t accessibleSize = 0;

    rtac::cuda::check_error( cudaGraphicsGLRegisterBuffer(
        &cudaResource, bufferId, cudaGraphicsRegisterFlagsWriteDiscard));
    rtac::cuda::check_error( cudaGraphicsMapResources(1, &cudaResource));
    
    rtac::cuda::check_error( cudaGraphicsResourceGetMappedPointer(
        &GLdevicePtr, &accessibleSize, cudaResource));
    
    if(accessibleSize < byteCount) {
        std::ostringstream oss;
        oss << "GL buffer not big enough for copy (needs " << byteCount 
            << "b, buffer is " << accessibleSize << "b)";
        throw std::runtime_error(oss.str());
    }

    rtac::cuda::check_error(cudaMemcpy(cudaDevicePtr, GLdevicePtr, byteCount,
                              cudaMemcpyDeviceToDevice));

    rtac::cuda::check_error( cudaGraphicsUnmapResources(1, &cudaResource));
    rtac::cuda::check_error( cudaGraphicsUnregisterResource(cudaResource));
}

}; //namespace cuda
}; //namespace display
}; //namespace rtac
