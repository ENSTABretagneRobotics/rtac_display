#ifndef _DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_

#include <iostream>

#include <rtac_base/types/Pose.h>
#include <rtac_base/geometry.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class View3DObj : public ViewObj
{
    public:

    using Mat4    = ViewObj::Mat4;
    using Shape   = ViewObj::Mat4;
    using Pose    = rtac::types::Pose<float>;
    using Vector3 = rtac::types::Vector3<float>;

    protected:

    Mat4 viewMatrix_;

    public:
    
    View3DObj(const Pose& pose = Pose(), const Mat4& projection = Mat4::Identity());

    void set_pose(const Pose& pose);
    void look_at(const Vector3& target);
    void look_at(const Vector3& target, const Vector3& position,
                 const Vector3& up = {0.0f,0.0f,1.0f});
                 

    virtual Mat4 view_matrix() const;
    Pose pose() const;
};
using View3D = Handle<View3DObj>;

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_
