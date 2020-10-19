#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

View::Ptr View::New(const Mat4& mat)
{
    return Ptr(new View(mat));
}

View::View(const Mat4& mat) :
    screenSize_({1,1}),
    projectionMatrix_(mat)
{}

void View::update_projection()
{}

void View::set_screen_size(const Shape& screen)
{
    screenSize_ = screen;
    this->update_projection();
}

View::Mat4 View::projection_matrix() const
{
    return projectionMatrix_;
}

View::Mat4 View::view_matrix() const
{
    return projectionMatrix_;
}

}; //namespace display
}; //namespace optix_helpers

