#ifndef _DEF_RTAC_DISPLAY_VIEW_3D_H_
#define _DEF_RTAC_DISPLAY_VIEW_3D_H_

#include <iostream>
#include <memory>

#include <rtac_base/types/Pose.h>
#include <rtac_base/geometry.h>

#include <rtac_display/views/View.h>

namespace rtac { namespace display {

/**
 * Handle point of view in a 3D scene. (This does **not** handle perspective
 * projection. For perspective projection see PinholeView instead).
 *
 * This mostly handles conversion from the RTAC framework 3D coordinates
 * conventions (x to right of screen, y towards back of screen, z up) to OpenGL
 * 3D conventions (x towards right of screen, y towards up, z towards front of
 * screen). It also implements some helpers such as setting the point of view
 * from a rtac::Pose object and other geometrical helpers.
 */
class View3D : public View
{
    public:

    using Ptr      = std::shared_ptr<View3D>;
    using ConstPtr = std::shared_ptr<const View3D>;

    using Mat4    = View::Mat4;
    using Shape   = View::Shape;
    using Pose    = rtac::Pose<float>;
    using Vec3    = rtac::Vector3<float>;
    using Vector3 = rtac::Vector3<float>;

    static const Mat4 viewFrameGL;

    protected:

    Mat4 viewMatrix_;

    public:
    
    static Ptr New(const Pose& pose = Pose::Identity(), const Mat4& projection = Mat4::Identity());
    static Ptr Create(const Pose& pose = Pose::Identity(), const Mat4& projection = Mat4::Identity());

    View3D(const Pose& pose = Pose::Identity(), const Mat4& projection = Mat4::Identity());

    void set_pose(const Pose& pose);
    void look_at(const Vector3& target);
    void look_at(const Vector3& target, const Vector3& position,
                 const Vector3& up = {0.0f,0.0f,1.0f});

    Mat4 raw_view_matrix() const;
    virtual Mat4 view_matrix() const;
    Pose pose() const;
    Vec3 translation() const { return viewMatrix_(Eigen::seqN(0,3),3); }

    void set_raw_view(const Mat4& viewMatrix);
};

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_VIEW_3D_H_
