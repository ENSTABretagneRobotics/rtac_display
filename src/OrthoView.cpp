#include <optix_helpers/display/OrthoView.h>

namespace optix_helpers { namespace display {

OrthoView::Ptr OrthoView::New(const Bounds& bounds,
                              const Pose& pose,
                              float zNear, float zFar)
{
    return Ptr(new OrthoView(bounds, pose, zNear, zFar));
}

OrthoView::OrthoView(const Bounds& bounds,
                           const Pose& pose,
                           float zNear, float zFar) :
    View3D(pose),
    bounds_(bounds),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

void OrthoView::update_projection()
{
    projectionMatrix_ = Mat4::Identity();

    projectionMatrix_(0,0) = 2.0f / (bounds_.right - bounds_.left);
    projectionMatrix_(1,1) = 2.0f / (bounds_.top   - bounds_.bottom);
    projectionMatrix_(2,2) = 2.0f / (zNear_ - zFar_);

    projectionMatrix_(0,3) = -(projectionMatrix_(0,0)*bounds_.left   + 1.0f);
    projectionMatrix_(1,3) = -(projectionMatrix_(1,1)*bounds_.bottom + 1.0f);
    projectionMatrix_(2,3) = projectionMatrix_(2,2)*zNear_ - 1.0f;
}

void OrthoView::set_bounds(const Bounds& bounds)
{
    bounds_ = bounds;
    this->update_projection();
}

void OrthoView::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

OrthoView::Bounds OrthoView::bounds() const
{
    return bounds_;
}

}; //namespace display
}; //namespace optix_helpers

