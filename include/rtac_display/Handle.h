#ifndef _DEF_RTAC_DISPLAY_HANDLE_H_
#define _DEF_RTAC_DISPLAY_HANDLE_H_

#include <memory>

namespace rtac { namespace display {

template <typename T>
using Handle = std::shared_ptr<T>;

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_HANDLE_H_
