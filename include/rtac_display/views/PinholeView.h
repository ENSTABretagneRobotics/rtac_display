#ifndef _DEF_RTAC_DISPLAY_PINHOLE_VIEW_H_
#define _DEF_RTAC_DISPLAY_PINHOLE_VIEW_H_

#include <iostream>
#include <cmath>

#include <rtac_base/types/Handle.h>

#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

class PinholeView : public View3D
{
    public:

    // Alignment issue (caused by integration of pcl, activation of vectorization)
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr      = rtac::types::Handle<PinholeView>;
    using ConstPtr = rtac::types::Handle<const PinholeView>;

    using Mat4    = View3D::Mat4;
    using Shape   = View3D::Mat4;
    using Pose    = View3D::Pose;
    using Vector3 = View3D::Vector3;

    protected:

    float fovy_;
    float zNear_;
    float zFar_;
    virtual void update_projection();

    public:
    
    static Ptr New(float fovy = 90.0f, const Pose& pose = Pose(),
                   float zNear = 0.1f, float zFar = 1000.0f);

    PinholeView(float fovy = 90.0f, const Pose& pose = Pose(),
                float zNear = 0.1f, float zFar = 1000.0f);

    void set_fovy(float fovy);
    void set_range(float zNear, float zFar);

    float fovy() const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_PINHOLE_VIEW_H_
