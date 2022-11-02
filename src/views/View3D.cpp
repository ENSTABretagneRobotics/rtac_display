#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

using namespace rtac::types::indexing;

/**
 * Homogeneous transformation matrix from RTAC screen coordinates to OpenGL
 * screen coordinates.
 */
const View3D::Mat4 View3D::viewFrameGL = (Mat4() << 1, 0, 0, 0,
                                                    0, 0,-1, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 0, 1).finished();

/**
 * Creates a new instance of View3D allocated on the heap
 *
 * @param pose       a rtac::types::Pose giving the position and the
 *                   orientation of the point of view in 3D space (RTAC
 *                   convention). Default is pose at origin and back of screen
 *                   towards global y.
 * @param projection a homogeneous matrix modeling the projection form 3D space
 *                   to 2D screen coordinates (OpenGL convention).
 *
 * @return a shared pointer to the new View3D instance.
 */
View3D::Ptr View3D::New(const Pose& pose, const Mat4& projection)
{
    return Ptr(new View3D(pose, projection));
}

/**
 * Creates a new instance of View3D allocated on the heap
 *
 * @param pose       a rtac::types::Pose giving the position and the
 *                   orientation of the point of view in 3D space (RTAC
 *                   convention). Default is pose at origin and back of screen
 *                   towards global y.
 * @param projection a homogeneous matrix modeling the projection form 3D space
 *                   to 2D screen coordinates (OpenGL convention).
 *
 * @return a shared pointer to the new View3D instance.
 */
View3D::Ptr View3D::Create(const Pose& pose, const Mat4& projection)
{
    return Ptr(new View3D(pose, projection));
}

/**
 * Constructor of View3D.
 *
 * @param pose       a rtac::types::Pose giving the position and the
 *                   orientation of the point of view in 3D space (RTAC
 *                   convention). Default is pose at origin and back of screen
 *                   towards global y.
 * @param projection a homogeneous matrix modeling the projection form 3D space
 *                   to 2D screen coordinates (OpenGL convention).
 */
View3D::View3D(const Pose& pose, const Mat4& projection) :
    View(projection)
{
    this->set_pose(pose);
}

/**
 * Change point of view in the 3D scene.
 *
 * Does not change the projection matrix.
 * 
 * @param pose a rtac::types::Pose giving the position and the orientation of
 *             the point of view in 3D space (RTAC convention). Default is
 *             pose at origin and back of screen towards global y.
 */
void View3D::set_pose(const Pose& pose)
{
    viewMatrix_ = pose.homogeneous_matrix() * viewFrameGL;
}

/**
 * Change point of view in the 3D scene to look towards a specific position.
 *
 * Does not change either the projection or the position of the point of
 * view.
 * 
 * @param target the position of the point to look at.
 */
void View3D::look_at(const Vector3& target)
{
    this->look_at(target, viewMatrix_(seqN(0,3),3));
}

/**
 * Change point of view in the 3D scene to look towards a specific position.
 *
 * Does not change the projection.
 * 
 * @param target   the position of the point to look at.
 * @param position new position of the point of view.
 * @param up       general direction of the "top" of the screen in global 3D
 *                 space.
 */
void View3D::look_at(const Vector3& target, const Vector3& position,
                     const Vector3& up)
{
    this->set_pose(Pose().look_at(target, position, up));
}

/**
 * To be removed ?
 */
View3D::Mat4 View3D::raw_view_matrix() const
{
    return viewMatrix_;
}

View3D::Mat4 View3D::view_matrix() const
{
    return projectionMatrix_ * viewMatrix_.inverse();
}

/**
 * @return a rtac::types::Pose representing the point of view position and
 *         orientation.
 */
View3D::Pose View3D::pose() const
{
    return Pose::from_homogeneous_matrix(viewMatrix_ * viewFrameGL.inverse());
}

void View3D::set_raw_view(const Mat4& viewMatrix)
{
    viewMatrix_ = viewMatrix;
}

}; //namespace display
}; //namespace rtac

