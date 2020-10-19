#ifndef _DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

class MeshRendererObj : public RendererObj
{
    public:
    
    using Mat4 = View3DObj::Mat4;
    using Mesh = rtac::types::Mesh<float,uint32_t,3>;
    using Pose    = View3DObj::Pose;
    using Color   = std::array<float,3>;

    protected:

    static const std::string vertexShader;
    static const std::string fragmentShader;

    size_t numPoints_;
    GLuint points_;
    GLuint normals_;
    Pose   pose_;
    Color  color_;

    protected:

    void allocate_points(size_t numPoints);
    void delete_points();

    public:

    MeshRendererObj(const View3D& view,
                    const Color& color = {1.0,1.0,1.0});

    void set_mesh(const Mesh& mesh);
    void set_pose(const Pose& pose);
    void set_color(const Color& color);

    virtual void draw();
};
using MeshRenderer = Handle<MeshRendererObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_
