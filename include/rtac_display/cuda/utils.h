#ifndef _DEF_RTAC_DISPLAY_CUDA_UTILS_H_
#define _DEF_RTAC_DISPLAY_CUDA_UTILS_H_

#include <iostream>
#include <sstream>

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display { namespace cuda {

void copy_to_gl(GLuint bufferId, const void* cudaDevicePtr, size_t byteCount);
void copy_from_gl(void* cudaDevicePtr, GLuint bufferId, size_t byteCount);

}; //namespace cuda
}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_CUDA_UTILS_H_
