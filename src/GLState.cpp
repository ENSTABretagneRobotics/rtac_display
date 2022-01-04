#include <rtac_display/GLState.h>

namespace rtac { namespace display {

GLState::GLState()
{
    for(auto key : StateNames) {
        stateMap_[key] = false;
    }
    for(auto key : DefaultTrueStates) {
        stateMap_[key] = true;
    }
}

}; //namespace display
}; //namespace rtac
