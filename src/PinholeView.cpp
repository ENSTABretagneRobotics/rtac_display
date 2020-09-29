#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

PinholeViewObj::PinholeViewObj(float fovy, const Pose& pose,
                               float zNear, float zFar) :
    View3DObj(pose),
    fovy_(fovy),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

void PinholeViewObj::update_projection()
{

}

void PinholeViewObj::set_fovy(float fovy)
{
    fovy_ = fovy;
    this->update_projection();
}

void PinholeViewObj::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

float PinholeViewObj::fovy() const
{
    return fovy_;
}

}; //namespace display
}; //namespace optix_helpers

