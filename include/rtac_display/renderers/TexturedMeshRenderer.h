#ifndef _DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_
#define _DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_

#include <rtac_base/types/common.h>
#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Point.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

template <typename Tp = types::Point3<float>,
          typename Tf = types::Point3<uint32_t>,
          typename Tu = types::Point2<float>>
class TexturedMeshRenderer : public Renderer
{
    public:

    using Ptr      = types::Handle<TexturedMeshRenderer<Tp,Tf,Tu>>;
    using ConstPtr = types::Handle<const TexturedMeshRenderer<Tp,Tf,Tu>>;

    using Mat4  = View3D::Mat4;
    using Pose  = View3D::Pose;
    using Color = std::array<float,4>;

    using Points = GLVector<Tp>;
    using Faces  = GLVector<Tf>;
    using UVs    = GLVector<Tu>;

    protected:

    static const std::string vertexShader;
    static const std::string fragmentShader;
    static const std::string vertexShaderTextured;
    static const std::string fragmentShaderTextured;

    GLuint solidRender_;
    GLuint texturedRender_;

    typename Points::Ptr points_;
    typename Faces::Ptr  faces_;
    typename UVs::Ptr    uvs_; // texture coordinates
    GLTexture::Ptr       texture_;

    Color color_;
    Pose  pose_;

    void draw_solid() const;
    void draw_textured() const;

    TexturedMeshRenderer(const View3D::Ptr& view);

    public:

    static Ptr New(const View3D::Ptr& view);
    
    typename Points::Ptr&     points()       { return points_; }
    typename Points::ConstPtr points() const { return points_; };
    typename Faces::Ptr&     faces()       { return faces_; }
    typename Faces::ConstPtr faces() const { return faces_; };
    typename UVs::Ptr&     uvs()       { return uvs_; }
    typename UVs::ConstPtr uvs() const { return uvs_; };
    GLTexture::Ptr&     texture()       { return texture_; }
    GLTexture::ConstPtr texture() const { return texture_; }

    void set_pose(const Color& color);
    void set_pose(const Pose& pose);

    virtual void draw();
};

template <typename Tp, typename Tf, typename Tu>
const std::string TexturedMeshRenderer<Tp,Tf,Tu>::vertexShader = std::string( R"(
#version 430 core

in vec3 point;

uniform mat4 view;
uniform mat4 projection;
uniform vec4 solidColor;

out vec4 c;

void main()
{
    gl_Position = projection*view*vec4(point, 1.0f);
    c = solidColor;
}
)");

template <typename Tp, typename Tf, typename Tu>
const std::string TexturedMeshRenderer<Tp,Tf,Tu>::fragmentShader = std::string( R"(
#version 430 core

in  vec4 c;
out vec4 outColor;

void main()
{
    outColor = c;
}
)");

template <typename Tp, typename Tf, typename Tu>
const std::string TexturedMeshRenderer<Tp,Tf,Tu>::vertexShaderTextured = std::string( R"(
#version 430 core

in vec3 point;
in vec2 uvIn;

uniform mat4 view;
uniform mat4 projection;

out vec2 uv;

void main()
{
    gl_Position = projection*view*vec4(point, 1.0f);
    uv = uvIn;
}
)");

template <typename Tp, typename Tf, typename Tu>
const std::string TexturedMeshRenderer<Tp,Tf,Tu>::fragmentShaderTextured = std::string( R"(
#version 430 core

in  vec2 uv;
out vec4 outColor;

uniform sampler2D texIn;

void main()
{
    outColor = texture(texIn, uv);
}
)");

template <typename Tp, typename Tf, typename Tu>
TexturedMeshRenderer<Tp,Tf,Tu>::TexturedMeshRenderer(const View3D::Ptr& view) :
    Renderer(vertexShader, fragmentShader, view),
    solidRender_(this->renderProgram_),
    texturedRender_(create_render_program(vertexShaderTextured, fragmentShaderTextured)),
    points_(new Points(0)),
    faces_(new Faces(0)),
    uvs_(new UVs(0)),
    texture_(GLTexture::New()),
    color_({1,1,0,1})
{}

template <typename Tp, typename Tf, typename Tu>
typename TexturedMeshRenderer<Tp,Tf,Tu>::Ptr
TexturedMeshRenderer<Tp,Tf,Tu>::New(const View3D::Ptr& view)
{
    return Ptr(new TexturedMeshRenderer<Tp,Tf,Tu>(view));
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::set_pose(const Pose& pose)
{
    pose_ = pose;
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw()
{
    if(!points_ || points_->size() == 0)
        return;

    GLfloat pointSize;
    glGetFloatv(GL_POINT_SIZE, &pointSize);
    glPointSize(3);

    glEnable(GL_DEPTH_TEST);

    if(!uvs_ || uvs_->size() == 0 || !texture_) {
        this->draw_solid();
    }
    else {
        this->draw_textured();
    }

    glPointSize(pointSize);
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw_solid() const
{
    glUseProgram(solidRender_);
    
    glBindBuffer(GL_ARRAY_BUFFER, points_->gl_id());
    glVertexAttribPointer(0, GLFormat<Tp>::Size, GLFormat<Tp>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    auto view = std::dynamic_pointer_cast<View3D>(view_);
    Mat4 viewMatrix = (view->raw_view_matrix().inverse()) * pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "projection"),
        1, GL_FALSE, view->projection_matrix().data());
    glUniform4fv(glGetUniformLocation(solidRender_, "solidColor"), 1, color_.data());
    
    if(faces_->size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_->gl_id());
        glDrawElements(GL_TRIANGLES, GLFormat<Tf>::Size*faces_->size(),
                       GLFormat<Tf>::Type, 0);
        // glDrawElements(GL_LINE_STRIP, GLFormat<Tf>::Size*faces_->size(),
        //                GLFormat<Tf>::Type, 0);
    }
    else {
        glDrawArrays(GL_TRIANGLES, 0, points_->size());
    }

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw_textured() const
{
    glUseProgram(texturedRender_);
    
    glBindBuffer(GL_ARRAY_BUFFER, points_->gl_id());
    glVertexAttribPointer(0, GLFormat<Tp>::Size, GLFormat<Tp>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, uvs_->gl_id());
    glVertexAttribPointer(1, GLFormat<Tu>::Size, GLFormat<Tu>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    
    auto view = std::dynamic_pointer_cast<View3D>(view_);
    Mat4 viewMatrix = (view->raw_view_matrix().inverse()) * pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(texturedRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniformMatrix4fv(glGetUniformLocation(texturedRender_, "projection"),
        1, GL_FALSE, view->projection_matrix().data());
    
    glUniform1i(glGetUniformLocation(texturedRender_, "texIn"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());

    if(faces_->size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_->gl_id());
        glDrawElements(GL_TRIANGLES, GLFormat<Tf>::Size*faces_->size(),
                       GLFormat<Tf>::Type, 0);
        // glDrawElements(GL_LINE_STRIP, GLFormat<Tf>::Size*faces_->size(),
        //                GLFormat<Tf>::Type, 0);
    }
    else {
        glDrawArrays(GL_TRIANGLES, 0, points_->size());
    }

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_
