#ifndef _DEF_RTAC_DISPLAY_RENDERERS_POINTCLOUD_RENDERER_H_
#define _DEF_RTAC_DISPLAY_RENDERERS_POINTCLOUD_RENDERER_H_

#include <memory>

#include <rtac_display/GLMesh.h>
#include <rtac_display/renderers/MeshRenderer.h>

namespace rtac { namespace display {

class PointCloudRenderer : public MeshRenderer
{
    public:

    using Ptr      = std::shared_ptr<PointCloudRenderer>;
    using ConstPtr = std::shared_ptr<const PointCloudRenderer>;

    protected:

    GLMesh::Ptr mutableMesh_;

    // Making render mode protected
    using MeshRenderer::set_render_mode;

    PointCloudRenderer(const GLContext::Ptr& context,
                       const Color::RGBAf& color = {1.0,1.0,1.0,1.0}) :
        MeshRenderer(context, color),
        mutableMesh_(GLMesh::Create())
    {
        mesh_ = mutableMesh_;
        this->set_render_mode(MeshRenderer::Mode::Points);
    }

    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const Color::RGBAf& color = {1.0,1.0,1.0,1.0})
    {
        return Ptr(new PointCloudRenderer(context, color));
    }

    GLMesh::Ptr mesh() { return mutableMesh_; }

    GLVector<GLMesh::Point>&       points()       { return mutableMesh_->points(); }
    const GLVector<GLMesh::Point>& points() const { return mutableMesh_->points(); }
};


}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDERERS_POINTCLOUD_RENDERER_H_
