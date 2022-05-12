#ifndef _DEF_RTAC_DISPLAY_RENDERER_FRAME_H_
#define _DEF_RTAC_DISPLAY_RENDERER_FRAME_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLContext.h>
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

    using Ptr      = rtac::types::Handle<Frame>;
    using ConstPtr = rtac::types::Handle<const Frame>;

    using Pose = View3D::Pose;

    protected:

    View3D::Pose pose_;

    Frame(const GLContext::Ptr& context,
          const View3D::Pose& pose = View3D::Pose(),
          const View::Ptr& view = View::New());

    Frame(const View3D::Pose& pose = View3D::Pose(),
          const View::Ptr& view = View::New());
    
    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const View3D::Pose& pose = View3D::Pose(),
                      const View::Ptr& view = View::New());

    static Ptr New(const View3D::Pose& pose = View3D::Pose(),
                   const View::Ptr& view = View::New());

    void set_pose(const View3D::Pose& pose);

    virtual void draw() const;
    virtual void draw(const View::ConstPtr& view) const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDERER_FRAME_H_

