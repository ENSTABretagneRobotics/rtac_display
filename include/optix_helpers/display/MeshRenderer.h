#ifndef _DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Source.h>
#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {

class MeshRendererObj : public RendererObj
{
    public:
    
    using Mesh = rtac::types::Mesh<float,uint32_t,3>;

    protected:

    static const Source vertexShader;
    static const Source fragmentShader;

    GLuint points_;
    GLuint faces_;
    GLuint normals_;

    public:

    MeshRendererObj(const View3D& view);

    virtual void draw();
};
using MeshRenderer = Handle<MeshRendererObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_MESH_RENDERER_H_
