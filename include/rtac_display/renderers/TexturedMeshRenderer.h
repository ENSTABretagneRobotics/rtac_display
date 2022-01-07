#ifndef _DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_
#define _DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_

#include <algorithm>

#include <rtac_base/types/common.h>
#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Point.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLContext.h>
#include <rtac_display/Color.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

/**
 * Displays a Mesh optionally textured.
 *
 * This will draw a 3D mesh. Depending on the provided information (faces,
 * texture coordinates, texture...) rendering type will be either a
 * solid-colored wire mesh or a textured mesh.
 */
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

    Color::RGBAf color_;
    Pose         pose_;

    void draw_solid(const View::ConstPtr& view) const;
    void draw_textured(const View::ConstPtr& view) const;

    TexturedMeshRenderer(const GLContext::Ptr& context,
                         const View3D::Ptr& view);
    TexturedMeshRenderer(const View3D::Ptr& view);

    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const View3D::Ptr& view);
    static Ptr New(const View3D::Ptr& view);
    
    typename Points::Ptr&     points()       { return points_; }
    typename Points::ConstPtr points() const { return points_; };
    typename Faces::Ptr&     faces()       { return faces_; }
    typename Faces::ConstPtr faces() const { return faces_; };
    typename UVs::Ptr&     uvs()       { return uvs_; }
    typename UVs::ConstPtr uvs() const { return uvs_; };
    GLTexture::Ptr&     texture()       { return texture_; }
    GLTexture::ConstPtr texture() const { return texture_; }

    void set_color(const Color::RGBAf& color);
    void set_pose(const Pose& pose);

    virtual void draw();
    virtual void draw(const View::ConstPtr& view);

    static Ptr from_ply(const std::string& path, const View3D::Ptr& view,
                        bool transposeUVs = false);
};

/**
 * Vertex shader for solid color renderering. Simply pass along the color to
 * the fragment shader.
 */
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

/**
 * Fragment shader for solid color renderering. Simply outputs the input color.
 */
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

/**
 * Vertex shader for textured renderering. Simply pass along the texture
 * coordinates to the fragment shader.
 */
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

/**
 * Fragment shader for textured rendering. Simply fetch color data from texture
 * using the textured coordinates passed along by the vertex shader.
 */
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

/**
 * Constructor of TexturedMeshRenderer.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param view a View instance to render this object.
 */
template <typename Tp, typename Tf, typename Tu>
TexturedMeshRenderer<Tp,Tf,Tu>::TexturedMeshRenderer(const GLContext::Ptr& context,
                                                     const View3D::Ptr& view) :
    Renderer(context, vertexShader, fragmentShader, view),
    solidRender_(this->renderProgram_),
    texturedRender_(create_render_program(vertexShaderTextured, fragmentShaderTextured)),
    points_(new Points(0)),
    faces_(new Faces(0)),
    uvs_(new UVs(0)),
    texture_(GLTexture::New()),
    color_({1,1,0,1})
{}

/**
 * Creates a new TexturedMeshRenderer object on the heap and outputs a shared_ptr.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param view a View instance to render this object.
 *
  @return a shader pointer to the newly created instance.
 */
template <typename Tp, typename Tf, typename Tu>
typename TexturedMeshRenderer<Tp,Tf,Tu>::Ptr
TexturedMeshRenderer<Tp,Tf,Tu>::Create(const GLContext::Ptr& context,
                                       const View3D::Ptr& view)
{
    return Ptr(new TexturedMeshRenderer<Tp,Tf,Tu>(context, view));
}

/**
 * Constructor of TexturedMeshRenderer.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param view a View instance to render this object.
 */
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

/**
 * Creates a new TexturedMeshRenderer object on the heap and outputs a shared_ptr.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param view a View instance to render this object.
 *
  @return a shader pointer to the newly created instance.
 */
template <typename Tp, typename Tf, typename Tu>
typename TexturedMeshRenderer<Tp,Tf,Tu>::Ptr
TexturedMeshRenderer<Tp,Tf,Tu>::New(const View3D::Ptr& view)
{
    return Ptr(new TexturedMeshRenderer<Tp,Tf,Tu>(view));
}

/**
 * Sets the solid-color of the object. (Ignored at rendering if texture
 * information were provided)
 *
 * @param color a RGBA color.
 */
template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::set_color(const Color::RGBAf& color)
{
    color_ = color;
}

/**
 * Sets the position of the object in 3D-space.
 *
 * @param pose a rtac::types::Pose representing the full pose (position and
 *             orientation) in 3D space.
 */
template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::set_pose(const Pose& pose)
{
    pose_ = pose;
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw()
{
    this->draw(this->view());
}

/**
 * Main drawing function.
 *
 * Depending on the provided information, this will dispatch rendering to
 * either a textured rendering (if a texture and texture coordinates were
 * provided) or a solid-colored wired rendering.
 */
template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw(const View::ConstPtr& view)
{
    if(!points_ || points_->size() == 0)
        return;

    GLfloat pointSize;
    glGetFloatv(GL_POINT_SIZE, &pointSize);
    glPointSize(3);

    glEnable(GL_DEPTH_TEST);

    if(!uvs_ || uvs_->size() == 0 || !texture_) {
        this->draw_solid(view);
    }
    else {
        this->draw_textured(view);
    }

    glPointSize(pointSize);
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw_solid(const View::ConstPtr& view) const
{
    glUseProgram(solidRender_);
    
    glBindBuffer(GL_ARRAY_BUFFER, points_->gl_id());
    glVertexAttribPointer(0, GLFormat<Tp>::Size, GLFormat<Tp>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    auto view3d = std::dynamic_pointer_cast<const View3D>(view);
    Mat4 viewMatrix = (view3d->raw_view_matrix().inverse()) * pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "projection"),
        1, GL_FALSE, view3d->projection_matrix().data());
    glUniform4fv(glGetUniformLocation(solidRender_, "solidColor"), 1, 
                 reinterpret_cast<const float*>(&color_));
    
    if(faces_->size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_->gl_id());
        // glDrawElements(GL_TRIANGLES, GLFormat<Tf>::Size*faces_->size(),
        //                GLFormat<Tf>::Type, 0);
        glDrawElements(GL_LINE_STRIP, GLFormat<Tf>::Size*faces_->size(),
                       GLFormat<Tf>::Type, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    else {
        // glDrawArrays(GL_TRIANGLES, 0, points_->size());
        glDrawArrays(GL_LINE_STRIP, 0, points_->size());
    }

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

template <typename Tp, typename Tf, typename Tu>
void TexturedMeshRenderer<Tp,Tf,Tu>::draw_textured(const View::ConstPtr& view) const
{
    glUseProgram(texturedRender_);
    
    glBindBuffer(GL_ARRAY_BUFFER, points_->gl_id());
    glVertexAttribPointer(0, GLFormat<Tp>::Size, GLFormat<Tp>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, uvs_->gl_id());
    glVertexAttribPointer(1, GLFormat<Tu>::Size, GLFormat<Tu>::Type, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    
    auto view3d = std::dynamic_pointer_cast<const View3D>(view);
    Mat4 viewMatrix = (view3d->raw_view_matrix().inverse()) * pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(texturedRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniformMatrix4fv(glGetUniformLocation(texturedRender_, "projection"),
        1, GL_FALSE, view3d->projection_matrix().data());
    
    glUniform1i(glGetUniformLocation(texturedRender_, "texIn"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());

    if(faces_->size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_->gl_id());
        glDrawElements(GL_TRIANGLES, GLFormat<Tf>::Size*faces_->size(),
                       GLFormat<Tf>::Type, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    else {
        glDrawArrays(GL_TRIANGLES, 0, points_->size());
    }

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

/**
 * Reads mesh information from a .ply file and instanciate a new
 * TextureMeshRenderer.
 *
 * PLY file specifications :
 * - Vertex position element must be named "vertex" with "x","y","z"
 *   properties of type float.
 * - Face indexes element must be named "face", with a single list property
 *   named either "vertex_indices" or "vertex_index" of type uchar or uint (to
 *   be checked).
 * - Texture coordinates element must be named "texCoord" with "x","y"
 *   properties of type float.
 * - The PLY file can be either ascii or binary (provided the endianess is
 *   correct).
 *
 * An OpenGL context must have been created beforehand.
 * 
 * @param path   path of the PLY to be loaded.
 * @param view   View instance for this object to be renderered with.
 * @transposeUVs boolean. If true will swap x and y texture coordinates (usually
 *               to fix an image orientation.
 *
 * @return a shared pointer to the newly created TexturedMeshRenderer instance.
 */
template <typename Tp, typename Tf, typename Tu>
typename TexturedMeshRenderer<Tp,Tf,Tu>::Ptr
TexturedMeshRenderer<Tp,Tf,Tu>::from_ply(const std::string& path, const View3D::Ptr& view,
                                         bool transposeUVs)
{
    std::ifstream f(path, std::ios::binary | std::ios::in);
    if(!f.is_open()) {
        throw std::runtime_error(
            "PointCloud::from_ply : could not open file for reading " + path);
    }
    happly::PLYData data(f);
    
    if(!data.hasElement("vertex")) {
        throw std::runtime_error(
            "Invalid ply file : No vertex defined in \"" + path + "\"");
    }

    auto mesh = New(view);
    
    {
        // Loading vertices
        auto x = data.getElement("vertex").getProperty<float>("x");
        auto y = data.getElement("vertex").getProperty<float>("y");
        auto z = data.getElement("vertex").getProperty<float>("z");

        mesh->points()->resize(x.size());
        auto ptr = mesh->points()->map();
        Tp* data = ptr;
        for(int i = 0; i < x.size(); i++) {
            data[i].x = x[i];
            data[i].y = y[i];
            data[i].z = z[i];
        }
    }

    if(data.hasElement("face")) {
        // Loading faces
        std::vector<std::string> names({"vertex_indices", "vertex_index"});
        std::vector<std::vector<uint32_t>> f;
        for(auto& name : names) {
            try {
                f = data.getElement("face").getListPropertyAnySign<uint32_t>(name);
                break;
            }
            catch(const std::runtime_error& e) {
                // wrong face index name, trying another
            }
        }
        
        mesh->faces()->resize(f.size());
        auto ptr = mesh->faces()->map();
        Tf* data = ptr;
        for(int i = 0; i < f.size(); i++) {
            data[i].x = f[i][0];
            data[i].y = f[i][1];
            data[i].z = f[i][2];
        }
    }

    if(data.hasElement("texCoords")) {
        // Loading texture coordinates
        auto x = data.getElement("texCoords").getProperty<float>("x");
        auto y = data.getElement("texCoords").getProperty<float>("y");

        mesh->uvs()->resize(x.size());
        auto ptr = mesh->uvs()->map();
        Tu* data = ptr;
        if(transposeUVs) {
            for(int i = 0; i < x.size(); i++) {
                data[i].x = y[i];
                data[i].y = x[i];
            }
        }
        else {
            for(int i = 0; i < x.size(); i++) {
                data[i].x = x[i];
                data[i].y = y[i];
            }
        }
    }
    
    return mesh;
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXTURED_MESH_RENDERER_H_
