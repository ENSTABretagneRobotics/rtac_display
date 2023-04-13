#pragma once

#include <rtac_display/utils.h>
#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>

#include <rtac_base/cuda/CudaVector.h>

namespace rtac { namespace display {


void element(const GLVector<float>& data);

float sum(const GLVector<float>& data);

float sum(cuda::CudaVector<float>& data);

}; //namespace display
}; //namespace rtac



