#include "reductions.h"

#include <rtac_base/cuda/reductions.hcu>

namespace rtac { namespace display {

float sum(cuda::DeviceVector<float>& data)
{
    cuda::device::reduce(data.data(), data.data(), data.size());
    return 0.0f;
}

}; //namespace display
}; //namespace rtac

