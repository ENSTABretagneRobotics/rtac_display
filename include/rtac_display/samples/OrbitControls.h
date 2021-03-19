#ifndef _DEF_RTAC_DISPLAY_SAMPLES_ORBIT_CONTROL_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_ORBIT_CONTROL_H_

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/common.h>

#include <rtac_display/EventHandler.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display { namespace samples {

class OrbitControls : public EventHandler
{
    public:
    
    using Ptr           = rtac::types::Handle<OrbitControls>;
    using ConstPtr      = rtac::types::Handle<const OrbitControls>;
    using Vec3          = View3D::Vector3;
    using Pose          = View3D::Pose;
    using Mat3          = View3D::Pose::Mat3;
    using Quaternion    = View3D::Pose::Quaternion;
    using Mat4          = View3D::Mat4;
    using MousePosition = rtac::types::Vector2<double>;

    protected:
    
    View3D::Ptr view_;
    Mat3        viewFrame_; // defined by the up_ parameter
    bool        locked_;
    Vec3        target_;
    float       alpha_;

    float rho_;
    float theta_;
    float phi_;
    MousePosition lastMouse_;

    void lock_view();
    void unlock_view();

    OrbitControls(const View3D::Ptr& view,
                  const Vec3& target = {0,0,0},
                  const Vec3& up = {0,0,1});
    public:

    static Ptr Create(const View3D::Ptr& view,
                      const Vec3& target = {0,0,0},
                      const Vec3& up     = {0,0,1});

    virtual void mouse_position_callback(double x, double y);
    virtual void mouse_button_callback(int button, int action, int modes);
};

}; //namespace samples
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_ORBIT_CONTROL_H_
