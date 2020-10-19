#include <optix_helpers/display/Renderer.h>

namespace optix_helpers { namespace display {

const std::string Renderer::vertexShader = std::string( R"(
#version 430 core

in vec3 point;
in vec3 color;
uniform mat4 view;
out vec3 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    gl_Position.z = 1.0f;
    c = color;
}
)");

const std::string Renderer::fragmentShader = std::string(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)");

Renderer::Ptr Renderer::New(const std::string& vertexShader,
                            const std::string& fragmentShader,
                            const View::Ptr& view)
{
    return Ptr(new Renderer(vertexShader, fragmentShader, view));
}

Renderer::Renderer(const std::string& vertexShader, const std::string& fragmentShader,
                   const View::Ptr& view) :
    renderProgram_(create_render_program(vertexShader, fragmentShader)),
    view_(view)
{}

void Renderer::draw()
{
    float vertices[] = {0,0,0,
                        1,0,0,
                        0,0,0,
                        0,1,0,
                        0,0,0,
                        0,0,1};
    float colors[] = {1,0,0,
                      1,0,0,
                      0,1,0,
                      0,1,0,
                      0,0,1,
                      0,0,1};

    GLfloat lineWidth;
    glGetFloatv(GL_LINE_WIDTH, &lineWidth);
    glLineWidth(3);

    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, colors);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, view_->view_matrix().data());

    glDrawArrays(GL_LINES, 0, 6);
    
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);
    glLineWidth(lineWidth);
}

void Renderer::set_view(const View::Ptr& view) const
{
    view_ = view;
}

View::Ptr Renderer::view() const
{
    return view_;
}

}; //namespace display
}; //namespace optix_helpers

