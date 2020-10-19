#include <optix_helpers/display/PinholeView.h>

#include <cmath>

namespace optix_helpers { namespace display {

PinholeView::Ptr PinholeView::New(float fovy, const Pose& pose,
                                  float zNear, float zFar)
{
    return Ptr(new PinholeView(fovy, pose, zNear, zFar));
}

PinholeView::PinholeView(float fovy, const Pose& pose,
                               float zNear, float zFar) :
    View3D(pose),
    fovy_(fovy),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

void PinholeView::update_projection()
{
    projectionMatrix_ = Mat4::Zero();

    projectionMatrix_(0,0) = 1.0 / std::tan(0.5f*M_PI*fovy_/180.0);
    projectionMatrix_(1,1) = projectionMatrix_(0,0) * screenSize_.ratio<float>();
    projectionMatrix_(2,2) = (zFar_ + zNear_) / (zNear_ - zFar_);
    projectionMatrix_(2,3) = 2.0f*zFar_*zNear_ / (zNear_ - zFar_);
    projectionMatrix_(3,2) = -1.0f;
}

void PinholeView::set_fovy(float fovy)
{
    fovy_ = fovy;
    this->update_projection();
}

void PinholeView::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

float PinholeView::fovy() const
{
    return fovy_;
}

}; //namespace display
}; //namespace optix_helpers

