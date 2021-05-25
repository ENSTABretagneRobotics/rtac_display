#ifndef _DEF_RTAC_DISPLAY_ORTHO_VIEW_H_
#define _DEF_RTAC_DISPLAY_ORTHO_VIEW_H_

#include <iostream>
#include <cmath>

#include <rtac_base/types/Rectangle.h>
#include <rtac_base/types/Handle.h>

#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

/**
 * Subclass of View3D to handle orthographic projection in 3D space.
 *
 * An orthographic projection is a projection without perspective (equivalent
 * to a perspective projection from infinitely far and infinitely zoomed in, a
 * zero angle field of view). It is modeled as a rectangular cuboid
 * parametrized by [left-right], [bottom-top] and [near-far] clipping planes.
 *
 * This kind of projection is often used in CAD design softwares to avoid the
 * distorsion of perspective projection.
 */
class OrthoView : public View3D
{
    public:

    using Ptr      = rtac::types::Handle<OrthoView>;
    using ConstPtr = rtac::types::Handle<const OrthoView>;

    using Mat4    = View3D::Mat4;
    using Shape   = View3D::Mat4;
    using Pose    = View3D::Pose;
    using Vector3 = View3D::Vector3;
    using Bounds  = rtac::types::Rectangle<float>;

    protected:

    Bounds bounds_;
    float zNear_;
    float zFar_;
    virtual void update_projection();

    public:
    
    static Ptr New(const Bounds& bounds = Bounds({-1,1,-1,1}),
                   const Pose& pose = Pose(),
                   float zNear = 0.1f, float zFar = 1000.0f);

    OrthoView(const Bounds& bounds = Bounds({-1,1,-1,1}),
              const Pose& pose = Pose(),
              float zNear = 0.1f, float zFar = 1000.0f);

    void set_bounds(const Bounds& bounds);
    void set_range(float zNear, float zFar);
    
    Bounds bounds() const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_ORTHO_VIEW_H_
