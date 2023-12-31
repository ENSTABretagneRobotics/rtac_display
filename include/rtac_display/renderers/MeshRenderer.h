#ifndef _DEF_RTAC_DISPLAY_MESH_RENDERER_H_
#define _DEF_RTAC_DISPLAY_MESH_RENDERER_H_

#include <memory>

#include <rtac_base/types/common.h>

#include <rtac_display/GLContext.h>
#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>
#include <rtac_display/GLMesh.h>
#include <rtac_display/GLTexture.h>

namespace rtac { namespace display {

class MeshRenderer : public Renderer
{
    public:

    using Ptr      = std::shared_ptr<MeshRenderer>;
    using ConstPtr = std::shared_ptr<const MeshRenderer>;
    
    using Mat4  = View3D::Mat4;
    using Pose  = View3D::Pose;

    enum Mode {
        Points,
        Solid,
        WireFrame,
        NormalShading,
        Textured,
        TexturedNormal,
    };

    protected:

    static const std::string vertexShaderSolid;
    static const std::string vertexShaderNormals;
    static const std::string vertexShaderDisplayNormals;
    static const std::string fragmentShaderSolid;
    static const std::string vertexShaderTextured;
    static const std::string fragmentShaderTextured;
    static const std::string vertexShaderTexturedNormal;
    static const std::string fragmentShaderTexturedNormal;
    
    GLMesh::ConstPtr    mesh_;
    Pose                pose_;
    Color::RGBAf        color_;
    GLTexture::ConstPtr texture_;

    Mode   renderMode_;
    GLuint solidRender_;
    GLuint normalShading_;
    GLuint texturedShading_;
    GLuint texturedNormalShading_;

    bool         displayNormals_;
    GLuint       displayNormalsProgram_;
    Color::RGBAf normalsColor_;

    protected:

    MeshRenderer(const GLContext::Ptr& context,
                 const Color::RGBAf& color = {1.0,1.0,1.0,1.0});

    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const Color::RGBAf& color = {1.0,1.0,1.0,1.0});

    void set_color(const Color::RGBAf& color);
    void set_pose(const Pose& pose)                      { pose_ = pose; }
    void set_texture(const GLTexture::ConstPtr& texture) { texture_ = texture; }

    virtual void draw(const View::ConstPtr& view) const;
    void draw_solid(const View::ConstPtr& view, GLenum primitiveMode) const;
    void draw_normal_shading(const View::ConstPtr& view) const;
    void draw_textured(const View::ConstPtr& view) const;
    void draw_normals(const View::ConstPtr& view) const;
    void draw_textured_normal(const View::ConstPtr& view) const;

    GLMesh::ConstPtr  mesh() const { return mesh_; }
    GLMesh::ConstPtr& mesh()       { return mesh_; }

    void set_render_mode(Mode mode) { renderMode_ = mode; }
    void enable_normals_display()  { displayNormals_ = true; }
    void disable_normals_display() { displayNormals_ = false; }
    void set_normals_color(const Color::RGBAf& color) { normalsColor_ = color; }
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_MESH_RENDERER_H_
