#ifndef _DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_

#include <iostream>

#include <rtac_base/types/Pose.h>
#include <rtac_base/geometry.h>

#include <optix_helpers/display/Handle.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class View3D : public View
{
    public:

    // Alignment issue (caused by integration of pcl, activation of vectorization)
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr      = Handle<View3D>;
    using ConstPtr = Handle<const View3D>;

    using Mat4    = View::Mat4;
    using Shape   = View::Shape;
    using Pose    = rtac::types::Pose<float>;
    using Vector3 = rtac::types::Vector3<float>;

    static const Mat4 viewFrameGL;

    protected:

    Mat4 viewMatrix_;

    public:
    
    static Ptr New(const Pose& pose = Pose(), const Mat4& projection = Mat4::Identity());

    View3D(const Pose& pose = Pose(), const Mat4& projection = Mat4::Identity());

    void set_pose(const Pose& pose);
    void look_at(const Vector3& target);
    void look_at(const Vector3& target, const Vector3& position,
                 const Vector3& up = {0.0f,0.0f,1.0f});
                 

    Mat4 raw_view_matrix() const;
    virtual Mat4 view_matrix() const;
    Pose pose() const;
};

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_VIEW_3D_H_
