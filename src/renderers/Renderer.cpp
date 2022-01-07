#include <rtac_display/renderers/Renderer.h>

namespace rtac { namespace display {

/**
 * Simple GLSL Vertex shader. This shader pass the color c to the fragment
 * shader and multiplies a vector by the view matrix.
 */
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

/**
 * Simple GLSL Fragment shader. Simply outputs the color passed as parameter.
 */
const std::string Renderer::fragmentShader = std::string(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)");

/**
 * Instanciate a new Renderer on the heap.
 *
 * @param vertexShader   vertex shader source which will be used to create the
 *                       OpenGL render program for this object (default is
 *                       Renderer::vertexShader).
 * @param fragmentShader fragment shader source which will be used to create the
 *                       OpenGL render program for this object (default is
 *                       Renderer::fragmentShader).
 * @param view           View to be used to render this object. Default is
 *                       Identity (no geometric transformation for renderering
 *                       => rendering in the 2D x-y plane).
 *
 * @return a shared pointer to the newly instanciated Renderer.
 */
Renderer::Ptr Renderer::Create(const GLContext::Ptr& context,
                               const std::string& vertexShader,
                               const std::string& fragmentShader,
                               const View::Ptr& view)
{
    return Ptr(new Renderer(context, vertexShader, fragmentShader, view));
}

/**
 * Constructor of Renderer
 *
 * @param vertexShader   vertex shader source which will be used to create the
 *                       OpenGL render program for this object (default is
 *                       Renderer::vertexShader).
 * @param fragmentShader fragment shader source which will be used to create the
 *                       OpenGL render program for this object (default is
 *                       Renderer::fragmentShader).
 * @param view           View to be used to render this object. Default is
 *                       Identity (no geometric transformation for renderering
 *                       => rendering in the 2D x-y plane).
 */
Renderer::Renderer(const GLContext::Ptr& context,
                   const std::string& vertexShader,
                   const std::string& fragmentShader,
                   const View::Ptr& view) :
    context_(context),
    renderProgram_(create_render_program(vertexShader, fragmentShader)),
    view_(view)
{}

Renderer::Ptr Renderer::New(const std::string& vertexShader,
                            const std::string& fragmentShader,
                            const View::Ptr& view)
{
    return Ptr(new Renderer(nullptr, vertexShader, fragmentShader, view));
}

Renderer::Renderer(const std::string& vertexShader,
                   const std::string& fragmentShader,
                   const View::Ptr& view) :
    Renderer(nullptr, vertexShader, fragmentShader, view)
{}

void Renderer::draw() const
{
    if(!this->view()) {
        throw std::runtime_error("No view in renderer");
    }
    this->draw(this->view());
}

/**
 * Performs the OpenGL API calls to draw an object. By default this draws a XYZ
 * frame at the origin.
 */
void Renderer::draw(const View::ConstPtr& view) const
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
        1, GL_FALSE, view->view_matrix().data());

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
}; //namespace rtac

