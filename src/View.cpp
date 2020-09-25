#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

ViewObj::ViewObj(const Mat4& mat) :
    projectionMatrix_(mat)
{}


void ViewObj::update_projection(size_t screenWidth, size_t screenHeight)
{}

ViewObj::Mat4 ViewObj::view_matrix(size_t screenWidth, size_t screenHeight)
{
    this->update_projection(screenWidth, screenHeight);
    return this->view_matrix();
}

ViewObj::Mat4 ViewObj::view_matrix() const
{
    return projectionMatrix_;
}

View::View(const Mat4& mat) :
    Handle<ViewObj>(mat)
{}

View::View(const std::shared_ptr<ViewObj>& obj) :
    Handle<ViewObj>(obj)
{}
    

}; //namespace display
}; //namespace optix_helpers

