#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

ViewObj::ViewObj(const Mat4& mat) :
    screenSize_({1,1}),
    projectionMatrix_(mat)
{}


void ViewObj::update_projection()
{}

void ViewObj::set_screen_size(const Shape& screen)
{
    screenSize_ = screen;
    this->update_projection();
}

ViewObj::Mat4 ViewObj::view_matrix() const
{
    return projectionMatrix_;
}

}; //namespace display
}; //namespace optix_helpers

