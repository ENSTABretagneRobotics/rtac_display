#include <rtac_display/GLState.h>

namespace rtac { namespace display {

constexpr std::array<GLenum,2>  GLState::DefaultTrueStates;
constexpr std::array<GLenum,89> GLState::StateNames;

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
