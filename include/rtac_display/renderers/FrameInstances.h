#ifndef _DEF_RTAC_DISPLAY_RENDERER_FRAME_INSTANCES_H_
#define _DEF_RTAC_DISPLAY_RENDERER_FRAME_INSTANCES_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLContext.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

/**
 * Similar to base class Renderer, but draws a fram (x,y,z) at a specific
 * position.
 */
class FrameInstances : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<FrameInstances>;
    using ConstPtr = rtac::types::Handle<const FrameInstances>;

    using Pose = View3D::Pose;

    protected:

    static const std::string vertexShader;
    static const std::string fragmentShader;

    View3D::Pose                 globalPose_;
    std::vector<Pose::Mat4>      poses_;
    mutable GLVector<Pose::Mat4> deviceData_;

    FrameInstances(const GLContext::Ptr& context,
                   const View::Ptr& view = View::New(),
                   const View3D::Pose& pose = View3D::Pose());

    FrameInstances(const View::Ptr& view = View::New(),
                   const View3D::Pose& pose = View3D::Pose());
    
    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const View::Ptr& view = View::New(),
                      const View3D::Pose& pose = View3D::Pose());

    static Ptr New(const View::Ptr& view = View::New(),
                   const View3D::Pose& pose = View3D::Pose());

    void set_global_pose(const Pose& pose) { globalPose_ = pose; }
    void add_pose(const Pose& pose) { poses_.push_back(pose.homogeneous_matrix()); }

    virtual void draw() const;
    virtual void draw(const View::ConstPtr& view) const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDERER_FRAME_INSTANCES_H_

