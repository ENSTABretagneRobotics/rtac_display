#ifndef _DEF_OPTIX_HELPERS_DISPLAY_PINHOLE_VIEW_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_PINHOLE_VIEW_H_

#include <iostream>

#include <optix_helpers/Handle.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

class PinholeViewObj : public View3DObj
{
    public:

    using Mat4    = View3DObj::Mat4;
    using Shape   = View3DObj::Mat4;
    using Pose    = View3DObj::Pose;
    using Vector3 = View3DObj::Vector3;

    protected:

    float fovy_;
    float zNear_;
    float zFar_;
    virtual void update_projection();

    public:
    
    PinholeViewObj(float fovy = 90.0f, const Pose& pose = Pose(),
                   float zNear = 0.1f, float zFar = 1000.0f);

    void set_fovy(float fovy);
    void set_range(float zNear, float zFar);

    float fovy() const;
};
using PinholeView = Handle<PinholeViewObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_PINHOLE_VIEW_H_
