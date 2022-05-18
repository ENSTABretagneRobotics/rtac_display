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


const std::string MeshRenderer::vertexShaderDisplayNormals = std::string( R"(
#version 430 core

layout(location = 0) in vec3  point;
layout(location = 1) in vec3  n;
layout(location = 2) in float nLenght;

uniform mat4 view;
uniform vec4 color;

out vec4 c;

void main()
{
    gl_Position = view*vec4(point + nLenght*n, 1.0f);
    c = color;
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

const std::string MeshRenderer::vertexShaderTextured = std::string( R"(
#version 430 core

in vec3 point;
in vec2 uvIn;

uniform mat4 view;

out vec2 uv;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    uv = uvIn;
}
)");

const std::string MeshRenderer::fragmentShaderTextured = std::string(R"(
#version 430 core

in  vec2 uv;
out vec4 outColor;

uniform sampler2D texIn;

void main()
{
    outColor = texture(texIn, uv);
}
)");

const std::string MeshRenderer::vertexShaderTexturedNormal = std::string( R"(
#version 430 core

in vec3 point;
in vec3 n;
in vec2 uvIn;

uniform mat4 view;

out float c;
out vec2  uv;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = abs(normalize((view*vec4(n, 0.0f)).xyz).z);
    uv = uvIn;
}
)");

const std::string MeshRenderer::fragmentShaderTexturedNormal = std::string(R"(
#version 430 core

in  float c;
in  vec2  uv;
out vec4 outColor;

uniform sampler2D texIn;

void main()
{
    outColor = c*texture(texIn, uv);
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
    renderMode_(Mode::TexturedNormal),
    solidRender_(this->renderProgram_),
    normalShading_(create_render_program(vertexShaderNormals, fragmentShaderSolid)),
    texturedShading_(create_render_program(vertexShaderTextured, fragmentShaderTextured)),
    texturedNormalShading_(create_render_program(vertexShaderTexturedNormal,
                                                 fragmentShaderTexturedNormal)),
    displayNormals_(false),
    displayNormalsProgram_(create_render_program(vertexShaderDisplayNormals, fragmentShaderSolid)),
    normalsColor_({0.0f,0.0f,1.0f,1.0f})
{}

MeshRenderer::Ptr MeshRenderer::New(const View3D::Ptr& view, const Color::RGBAf& color)
{
    return Ptr(new MeshRenderer(view, color));
}

MeshRenderer::MeshRenderer(const View3D::Ptr& view, const Color::RGBAf& color) :
    Renderer(vertexShader, fragmentShader, view),
    color_(color),
    renderMode_(Mode::TexturedNormal),
    solidRender_(this->renderProgram_),
    normalShading_(create_render_program(vertexShaderNormals,  fragmentShaderSolid)),
    texturedShading_(create_render_program(vertexShaderTextured, fragmentShaderTextured)),
    texturedNormalShading_(create_render_program(vertexShaderTexturedNormal,
                                                 fragmentShaderTexturedNormal)),
    displayNormals_(false),
    displayNormalsProgram_(create_render_program(vertexShaderDisplayNormals, fragmentShaderSolid)),
    normalsColor_({0.0f,0.0f,1.0f,1.0f})
{}

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
            this->draw_textured(view);
            break;
        case Mode::TexturedNormal:
            this->draw_textured_normal(view);
            break;
    }

    if(displayNormals_)
        this->draw_normals(view);
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
    glUniformMatrix4fv(glGetUniformLocation(normalShading_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(normalShading_, "color"),
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

void MeshRenderer::draw_textured(const View::ConstPtr& view) const
{
    if(!texture_ || mesh_->uvs().size() != mesh_->points().size()) {
        this->draw_normal_shading(view);
        return;
    }
    glUseProgram(texturedShading_);
    
    mesh_->points().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    mesh_->uvs().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);

    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(texturedShading_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform1i(glGetUniformLocation(texturedShading_, "texIn"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());

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

void MeshRenderer::draw_textured_normal(const View::ConstPtr& view) const
{
    if(!texture_ || mesh_->uvs().size()     != mesh_->points().size()
                 || mesh_->normals().size() != mesh_->points().size()) {
        this->draw_textured(view);
        return;
    }
    glUseProgram(texturedNormalShading_);
    
    mesh_->points().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    mesh_->normals().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    mesh_->uvs().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(2);

    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(texturedNormalShading_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform1i(glGetUniformLocation(texturedNormalShading_, "texIn"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());

    if(mesh_->faces().size() == 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, mesh_->points().size());
    }
    else {
        mesh_->faces().bind(GL_ELEMENT_ARRAY_BUFFER);
        glDrawElements(GL_TRIANGLES, 3*mesh_->faces().size(), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

void MeshRenderer::draw_normals(const View::ConstPtr& view) const
{
    static constexpr const float nLength[2] = {0.1f,1.0f};

    if(mesh_->normals().size() != mesh_->points().size()) return;

    glUseProgram(displayNormalsProgram_);
    
    mesh_->points().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    mesh_->normals().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nLength);
    glEnableVertexAttribArray(2);

    glVertexAttribDivisor(0, 2);
    glVertexAttribDivisor(1, 2);

    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(displayNormalsProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(displayNormalsProgram_, "color"),
        1, reinterpret_cast<const float*>(&normalsColor_));
    
    glDrawArraysInstanced(GL_LINES, 0, 2, 2*mesh_->points().size());

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVertexAttribDivisor(0, 0);
    glVertexAttribDivisor(1, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

}; //namespace display
}; //namespace rtac


