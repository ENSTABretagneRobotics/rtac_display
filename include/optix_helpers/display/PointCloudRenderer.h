#ifndef _DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_

#include <iostream>
#include <array>
#include <algorithm>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {


class PointCloudRendererObj : public RendererObj
{
    public:

    using Mat4    = View3DObj::Mat4;
    using Shape   = View3DObj::Shape;
    using Pose    = View3DObj::Pose;
    using Color   = std::array<float,3>;

    static const Source vertexShader;
    static const Source fragmentShader;

    protected:
    
    size_t numPoints_;
    GLuint points_;
    Pose   position_;
    Color  color_;
    
    void allocate_points(size_t numPoints);
    void delete_points();

    public:
    
    PointCloudRendererObj(const View3D& view,
                          const Color& color = {0.7,0.7,1.0});
    ~PointCloudRendererObj();
    
    void set_points(size_t numPoints, const float* data);
    void set_points(const RenderBufferGL& buffer);
    void set_color(const Color& color);

    virtual void draw();
};
using PointCloudRenderer = Handle<PointCloudRendererObj>;


}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_
