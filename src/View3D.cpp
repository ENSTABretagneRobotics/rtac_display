#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

using namespace rtac::types::indexing;

const View3DObj::Mat4 View3DObj::viewFrameGL = (Mat4() << 1, 0, 0, 0,
                                                          0, 0,-1, 0,
                                                          0, 1, 0, 0,
                                                          0, 0, 0, 1).finished();

View3DObj::View3DObj(const Pose& pose, const Mat4& projection) :
    ViewObj(projection)
{
    this->set_pose(pose);
}

void View3DObj::set_pose(const Pose& pose)
{
    viewMatrix_ = pose.homogeneous_matrix() * viewFrameGL;
}

void View3DObj::look_at(const Vector3& target)
{
    this->look_at(target, viewMatrix_(seqN(0,3),3));
}

void View3DObj::look_at(const Vector3& target, const Vector3& position,
                        const Vector3& up)
{
    this->set_pose(rtac::geometry::look_at(target, position, up));
}

View3DObj::Mat4 View3DObj::raw_view_matrix() const
{
    return viewMatrix_.inverse();
}

View3DObj::Mat4 View3DObj::view_matrix() const
{
    return projectionMatrix_ * viewMatrix_.inverse();
}

View3DObj::Pose View3DObj::pose() const
{
    return Pose::from_homogeneous_matrix(viewMatrix_ * viewFrameGL.inverse());
}

}; //namespace display
}; //namespace optix_helpers

