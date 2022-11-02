#include <rtac_display/views/PinholeView.h>

namespace rtac { namespace display {

/**
 * Creates a new PinholeView instance on the heap.
 *
 * @param fovy  vertical field of view in degrees.
 * @param pose  position and orientation of the camera in 3D space.
 * @param zNear minimum "distance to screen" at which objects will be
 *              renderered. Any vertex below this distance to the camera will
 *              be discarded by OpenGL.
 * @param zFar  maximum "distance to screen" at which objects will be
 *              renderered. Any vertex above this distance to the camera will
 *              be discarded by OpenGL.
 *
 * @return a shared pointer to the newly created PinholeView instance.
 */
PinholeView::Ptr PinholeView::New(float fovy, const Pose& pose,
                                  float zNear, float zFar)
{
    return Ptr(new PinholeView(fovy, pose, zNear, zFar));
}

/**
 * Creates a new PinholeView instance on the heap.
 *
 * @param fovy  vertical field of view in degrees.
 * @param pose  position and orientation of the camera in 3D space.
 * @param zNear minimum "distance to screen" at which objects will be
 *              renderered. Any vertex below this distance to the camera will
 *              be discarded by OpenGL.
 * @param zFar  maximum "distance to screen" at which objects will be
 *              renderered. Any vertex above this distance to the camera will
 *              be discarded by OpenGL.
 *
 * @return a shared pointer to the newly created PinholeView instance.
 */
PinholeView::Ptr PinholeView::Create(float fovy, const Pose& pose,
                                     float zNear, float zFar)
{
    return Ptr(new PinholeView(fovy, pose, zNear, zFar));
}

/**
 * Constructor of PinholeView
 *
 * @param fovy  vertical field of view in degrees.
 * @param pose  position and orientation of the camera in 3D space.
 * @param zNear minimum "distance to screen" at which objects will be
 *              renderered. Any vertex below this distance to the camera will
 *              be discarded by OpenGL.
 * @param zFar  maximum "distance to screen" at which objects will be
 *              renderered. Any vertex above this distance to the camera will
 *              be discarded by OpenGL.
 */
PinholeView::PinholeView(float fovy, const Pose& pose,
                               float zNear, float zFar) :
    View3D(pose),
    fovy_(fovy),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

/**
 * Computes the perspective projection matrix from the fovy_, zNear, zFar and
 * View::screenSize_ attributes.
 *
 * More information on how the projection matrix is computed
 * [here](https://learnopengl.com/Getting-started/Coordinate-Systems).
 */
void PinholeView::update_projection()
{
    projectionMatrix_ = Mat4::Zero();

    projectionMatrix_(0,0) = 1.0 / std::tan(0.5f*M_PI*fovy_/180.0);
    projectionMatrix_(1,1) = projectionMatrix_(0,0) * screenSize_.ratio<float>();
    projectionMatrix_(2,2) = (zFar_ + zNear_) / (zNear_ - zFar_);
    projectionMatrix_(2,3) = 2.0f*zFar_*zNear_ / (zNear_ - zFar_);
    projectionMatrix_(3,2) = -1.0f;
}

/**
 * Set a new vertical field of view and recompute the projection matrix.
 *
 * @param fovy vertical field of view in degrees.
 */
void PinholeView::set_fovy(float fovy)
{
    fovy_ = fovy;
    this->update_projection();
}

/**
 * Set minimum and maximum viewing distance.
 *
 * @param zNear minimum displayable distance.
 * @param zFar  maximum displayable distance.
 */
void PinholeView::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

/**
 * @return vertical field of view in degree.
 */
float PinholeView::fovy() const
{
    return fovy_;
}

}; //namespace display
}; //namespace rtac

