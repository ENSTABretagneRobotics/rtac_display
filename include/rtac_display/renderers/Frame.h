#ifndef _DEF_RTAC_DISPLAY_RENDERER_FRAME_H_
#define _DEF_RTAC_DISPLAY_RENDERER_FRAME_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

/**
 * Similar to base class Renderer, but draws a fram (x,y,z) at a specific
 * position.
 */
class Frame : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<Renderer>;
    using ConstPtr = rtac::types::Handle<const Renderer>;

    using Pose = View3D::Pose;

    protected:

    View3D::Pose pose_;

    public:

    static Ptr New(const View3D::Pose& pose = View3D::Pose(),
                   const View::Ptr& view = View::New());

    Frame(const View3D::Pose& pose = View3D::Pose(),
          const View::Ptr& view = View::New());
    
    void set_pose(const View3D::Pose& pose);

    virtual void draw();
    virtual void draw(const View::Ptr& view);
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDERER_FRAME_H_

