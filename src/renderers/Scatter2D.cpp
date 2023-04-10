#include <rtac_display/renderers/Scatter2D.h>

#include <algorithm>

namespace rtac { namespace display {

const std::string Scatter2D::vertexShader = std::string( R"(
#version 430 core

in float x;
in float y;
uniform vec2  xScale;
uniform vec2  yScale;

void main()
{
    gl_Position = vec4(fma(xScale.x, x, xScale.y), 
                       fma(yScale.x, y, yScale.y),
                       0.0f, 1.0f);
}
)");

/**
 * Simple GLSL Fragment shader. Simply outputs the color passed as parameter.
 */
const std::string Scatter2D::fragmentShader = std::string(R"(
#version 430 core

uniform vec4 color;
out vec4 outColor;

void main()
{
    outColor = color;
}
)");

Scatter2D::Scatter2D(const GLContext::Ptr& context) :
    Renderer(context, vertexShader, fragmentShader),
    xRange_(0,1), yRange_(0,1),
    color_({0.0, 0.8, 1.0, 1.0})
{}

Scatter2D::Ptr Scatter2D::Create(const GLContext::Ptr& context)
{
    return Ptr(new Scatter2D(context));
}

void Scatter2D::update_range()
{
    xRange_.lower = reductor_.min(x_);
    xRange_.upper = reductor_.max(x_);
    yRange_.lower = reductor_.min(y_);
    yRange_.upper = reductor_.max(y_);
}

void Scatter2D::draw(const View::ConstPtr& view) const
{
    if(x_.size() == 0 || y_.size() != x_.size()) {
        return;
    }

    glPointSize(10);
    glUseProgram(this->renderProgram_);

    x_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    y_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    auto screen = view->screen_size();
    auto screenAspect = screen.ratio<float>();
    auto dataMax = std::max(std::max(std::abs(xRange_.lower), std::abs(xRange_.upper)),
                            std::max(std::abs(yRange_.lower), std::abs(yRange_.upper)));
    
    float ax = 1.0f / screenAspect, ay = 1;
    if(screenAspect < 1.0f) {
        ax = 1; ay = screenAspect;
    }

    glUniform2f(glGetUniformLocation(renderProgram_, "xScale"), ax, 0.0f);
    glUniform2f(glGetUniformLocation(renderProgram_, "yScale"), ay, 0.0f);

    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
                 1, (const float*)&color_);

    glDrawArrays(GL_POINTS, 0, x_.size());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}


} //namespace display
} //namespace rtac
