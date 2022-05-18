#include <rtac_display/renderers/MeshRenderer.h>

namespace rtac { namespace display {

const std::string MeshRenderer::vertexShaderSolid = std::string( R"(
#version 430 core

in vec3 point;

uniform mat4 view;
uniform vec4 color;

out vec4 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = color;
}
)");

const std::string MeshRenderer::vertexShaderNormals = std::string( R"(
#version 430 core

in vec3 point;
in vec3 n;

uniform mat4 view;
uniform vec4 color;

out vec4 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    vec3 tmp = normalize((view*vec4(n, 0.0f)).xyz);
    c = abs(tmp.z)*color;
    //c = 0.5f*(1.0f - tmp.z)*color;
}
)");

const std::string MeshRenderer::fragmentShaderSolid = std::string(R"(
#version 430 core

in vec4 c;
out vec4 outColor;

void main()
{
    outColor = c;
}
)");

MeshRenderer::Ptr MeshRenderer::Create(const GLContext::Ptr& context,
                                       const View3D::Ptr& view,
                                       const Color::RGBAf& color)
{
    return Ptr(new MeshRenderer(context, view, color));
}

MeshRenderer::MeshRenderer(const GLContext::Ptr& context,
                           const View3D::Ptr& view,
                           const Color::RGBAf& color) :
    Renderer(context, vertexShaderSolid, fragmentShaderSolid, view),
    color_(color),
    renderMode_(Mode::NormalShading),
    solidRender_(this->renderProgram_),
    normalShading_(create_render_program(vertexShaderNormals, fragmentShaderSolid))
{}

MeshRenderer::Ptr MeshRenderer::New(const View3D::Ptr& view, const Color::RGBAf& color)
{
    return Ptr(new MeshRenderer(view, color));
}

MeshRenderer::MeshRenderer(const View3D::Ptr& view, const Color::RGBAf& color) :
    Renderer(vertexShader, fragmentShader, view),
    color_(color),
    renderMode_(Mode::Points),
    solidRender_(this->renderProgram_),
    normalShading_(create_render_program(vertexShaderNormals, fragmentShaderSolid))
{}

void MeshRenderer::set_pose(const Pose& pose)
{
    pose_ = pose;
}

void MeshRenderer::set_color(const Color::RGBAf& color)
{
    color_.r = std::max(0.0f, std::min(1.0f, color.r));
    color_.g = std::max(0.0f, std::min(1.0f, color.g));
    color_.b = std::max(0.0f, std::min(1.0f, color.b));
    color_.a = std::max(0.0f, std::min(1.0f, color.a));
}

void MeshRenderer::draw() const
{
    if(!this->view()) {
        throw std::runtime_error("No view in renderer");
    }
    this->draw(this->view());
}

void MeshRenderer::draw(const View::ConstPtr& view) const
{
    if(!mesh_) return;

    switch(renderMode_) {
        default:
        case Mode::Points:
            this->draw_solid(view, GL_POINTS);
            break;
        case Mode::Solid:
            this->draw_solid(view, GL_TRIANGLES);
            break;
        case Mode::WireFrame:
            this->draw_solid(view, GL_LINES);
            break;
        case Mode::NormalShading:
            this->draw_normal_shading(view);
            break;
        case Mode::Textured:
            //this->draw_textured(view);
            break;
    }
}

void MeshRenderer::draw_solid(const View::ConstPtr& view, GLenum primitiveMode) const
{
    glUseProgram(solidRender_);
    
    mesh_->points().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(solidRender_, "color"),
        1, reinterpret_cast<const float*>(&color_));

    if(mesh_->faces().size() == 0 || primitiveMode == GL_POINTS) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDrawArrays(primitiveMode, 0, mesh_->points().size());
    }
    else {
        mesh_->faces().bind(GL_ELEMENT_ARRAY_BUFFER);
        glDrawElements(primitiveMode, 3*mesh_->faces().size(), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

void MeshRenderer::draw_normal_shading(const View::ConstPtr& view) const
{
    if(mesh_->normals().size() == 0) {
        this->draw_solid(view, GL_TRIANGLES);
        return;
    }
    glUseProgram(normalShading_);
    
    mesh_->points().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    mesh_->normals().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);

    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(solidRender_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(solidRender_, "color"),
        1, reinterpret_cast<const float*>(&color_));

    if(mesh_->faces().size() == 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, mesh_->points().size());
    }
    else {
        mesh_->faces().bind(GL_ELEMENT_ARRAY_BUFFER);
        glDrawElements(GL_TRIANGLES, 3*mesh_->faces().size(), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

}; //namespace display
}; //namespace rtac


