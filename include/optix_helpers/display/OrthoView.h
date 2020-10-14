#ifndef _DEF_OPTIX_HELPERS_DISPLAY_ORTHO_VIEW_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_ORTHO_VIEW_H_

#include <iostream>

#include <rtac_base/types/Rectangle.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

class OrthoViewObj : public View3DObj
{
    public:

    // Alignment issue (caused by integration of pcl, activation of vectorization)
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Mat4    = View3DObj::Mat4;
    using Shape   = View3DObj::Mat4;
    using Pose    = View3DObj::Pose;
    using Vector3 = View3DObj::Vector3;
    using Bounds  = rtac::types::Rectangle<float>;

    protected:

    Bounds bounds_;
    float zNear_;
    float zFar_;
    virtual void update_projection();

    public:
    
    OrthoViewObj(const Bounds& bounds = Bounds({-1,1,-1,1}),
                 const Pose& pose = Pose(),
                 float zNear = 0.1f, float zFar = 1000.0f);

    void set_bounds(const Bounds& bounds);
    void set_range(float zNear, float zFar);
    
    Bounds bounds() const;
};
using OrthoView = Handle<OrthoViewObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_ORTHO_VIEW_H_
