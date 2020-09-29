#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

ViewObj::ViewObj(const Mat4& mat) :
    projectionMatrix_(mat)
{}


void ViewObj::update_projection(const Shape& scree)
{}

ViewObj::Mat4 ViewObj::view_matrix(const Shape& screen)
{
    this->update_projection(screen);
    return this->view_matrix();
}

ViewObj::Mat4 ViewObj::view_matrix() const
{
    return projectionMatrix_;
}

}; //namespace display
}; //namespace optix_helpers

