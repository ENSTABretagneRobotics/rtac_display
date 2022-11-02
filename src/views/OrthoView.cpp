#include <rtac_display/views/OrthoView.h>

namespace rtac { namespace display {

/**
 * Create a new instance of OrthoView on the heap.
 *
 * @param bounds a rtac::types::Rectangle object representing the left-right,
 *               bottom-top clipping plane positions.
 * @param pose   3D position (translation and orientation) of the camera.
 * @param zNear  near clipping plane position (vertex below this distance to
 *               the camera will be clipped.
 * @param zFar   far clipping plane position (vertex above this distance to the
 *               camera will be clipped.
 *
 * @return a shared pointer to the newly created OrthoView instance.
 */
OrthoView::Ptr OrthoView::New(const Bounds& bounds,
                              const Pose& pose,
                              float zNear, float zFar)
{
    return Ptr(new OrthoView(bounds, pose, zNear, zFar));
}

/**
 * Create a new instance of OrthoView on the heap.
 *
 * @param bounds a rtac::types::Rectangle object representing the left-right,
 *               bottom-top clipping plane positions.
 * @param pose   3D position (translation and orientation) of the camera.
 * @param zNear  near clipping plane position (vertex below this distance to
 *               the camera will be clipped.
 * @param zFar   far clipping plane position (vertex above this distance to the
 *               camera will be clipped.
 *
 * @return a shared pointer to the newly created OrthoView instance.
 */
OrthoView::Ptr OrthoView::Create(const Bounds& bounds,
                                 const Pose& pose,
                                 float zNear, float zFar)
{
    return Ptr(new OrthoView(bounds, pose, zNear, zFar));
}

/**
 * Constructor of OrthoView.
 *
 * @param bounds a rtac::types::Rectangle object representing the left-right,
 *               bottom-top clipping plane positions.
 * @param pose   3D position (translation and orientation) of the camera.
 * @param zNear  near clipping plane position (vertex below this distance to
 *               the camera will be clipped.
 * @param zFar   far clipping plane position (vertex above this distance to the
 *               camera will be clipped.
 */
OrthoView::OrthoView(const Bounds& bounds,
                           const Pose& pose,
                           float zNear, float zFar) :
    View3D(pose),
    bounds_(bounds),
    zNear_(zNear),
    zFar_(zFar)
{
    this->update_projection();
}

/**
 * Computes the orthographic projection matrix from the bounds_, zNear, zFar
 * and View::screenSize_ attributes.
 *
 * More information on how the projection matrix is computed
 * [here](https://learnopengl.com/Getting-started/Coordinate-Systems).
 */
void OrthoView::update_projection()
{
    projectionMatrix_ = Mat4::Identity();

    projectionMatrix_(0,0) = 2.0f / (bounds_.right - bounds_.left);
    projectionMatrix_(1,1) = 2.0f / (bounds_.top   - bounds_.bottom);
    projectionMatrix_(2,2) = 2.0f / (zNear_ - zFar_);

    projectionMatrix_(0,3) = -(projectionMatrix_(0,0)*bounds_.left   + 1.0f);
    projectionMatrix_(1,3) = -(projectionMatrix_(1,1)*bounds_.bottom + 1.0f);
    projectionMatrix_(2,3) = projectionMatrix_(2,2)*zNear_ - 1.0f;
}

/**
 * Set the [left-right], [bottom-top] clipping planes.
 * 
 * @param bounds a rtac::types::Rectangle object representing the left-right,
 *               bottom-top clipping plane positions.
 */
void OrthoView::set_bounds(const Bounds& bounds)
{
    bounds_ = bounds;
    this->update_projection();
}

/**
 * Set the zNear, zFar clipping planes.
 *
 * @param zNear  near clipping plane position (vertex below this distance to
 *               the camera will be clipped.
 * @param zFar   far clipping plane position (vertex above this distance to the
 *               camera will be clipped.
 */
void OrthoView::set_range(float zNear, float zFar)
{
    zNear_ = zNear;
    zFar_  = zFar;
    this->update_projection();
}

/**
 * @return a rtac::types::Rectangle object representing the left-right,
 *         bottom-top clipping plane positions.
 */
OrthoView::Bounds OrthoView::bounds() const
{
    return bounds_;
}

}; //namespace display
}; //namespace rtac

