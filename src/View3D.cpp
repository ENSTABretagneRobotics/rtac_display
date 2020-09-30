#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

using namespace rtac::types::indexing;

View3DObj::View3DObj(const Pose& pose, const Mat4& projection) :
    ViewObj(projection),
    viewMatrix_(pose.homogeneous_matrix())
{}

void View3DObj::set_pose(const Pose& pose)
{
    viewMatrix_ = pose.homogeneous_matrix();
}

void View3DObj::look_at(const Vector3& target)
{
    this->look_at(target, viewMatrix_(seqN(0,3),3));
}

void View3DObj::look_at(const Vector3& target, const Vector3& position,
                        const Vector3& up)
{
    Mat4 glView;
    glView << 1,0, 0,0,
              0,0,-1,0,
              0,1, 0,0,
              0,0, 0,1;
    viewMatrix_ = rtac::geometry::look_at(target, position, up).homogeneous_matrix();
    viewMatrix_ = viewMatrix_ * glView;
}

View3DObj::Mat4 View3DObj::view_matrix() const
{
    return projectionMatrix_ * viewMatrix_.inverse();
}

View3DObj::Pose View3DObj::pose() const
{
    return Pose::from_homogeneous_matrix(viewMatrix_);
}

}; //namespace display
}; //namespace optix_helpers

