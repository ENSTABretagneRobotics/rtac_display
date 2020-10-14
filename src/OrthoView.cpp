#include <optix_helpers/display/OrthoView.h>

#include <cmath>

namespace optix_helpers { namespace display {

OrthoViewObj::OrthoViewObj(const Bounds& bounds,
                           const Pose& pose,
                           float zNear, float zFar) :
    View3DObj(pose),
    bounds_(bounds),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

void OrthoViewObj::update_projection()
{
    projectionMatrix_ = Mat4::Identity();

    projectionMatrix_(0,0) = 2.0f / (bounds_.right - bounds_.left);
    projectionMatrix_(1,1) = 2.0f / (bounds_.top   - bounds_.bottom);
    projectionMatrix_(2,2) = 2.0f / (zNear_ - zFar_);

    projectionMatrix_(0,3) = -(projectionMatrix_(0,0)*bounds_.left   + 1.0f);
    projectionMatrix_(1,3) = -(projectionMatrix_(1,1)*bounds_.bottom + 1.0f);
    projectionMatrix_(2,3) = projectionMatrix_(2,2)*zNear_ - 1.0f;
}

void OrthoViewObj::set_bounds(const Bounds& bounds)
{
    bounds_ = bounds;
    this->update_projection();
}

void OrthoViewObj::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

OrthoViewObj::Bounds OrthoViewObj::bounds() const
{
    return bounds_;
}

}; //namespace display
}; //namespace optix_helpers

