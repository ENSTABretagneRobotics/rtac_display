#ifndef _DEF_OPTIX_HELPERS_DISPLAY_HANDLE_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_HANDLE_H_

#include <memory>

namespace optix_helpers { namespace display {

template <typename T>
using Handle = std::shared_ptr<T>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_HANDLE_H_
