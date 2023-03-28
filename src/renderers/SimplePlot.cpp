#include <rtac_display/renderers/SimplePlot.h>

namespace rtac { namespace display {

const std::string SimplePlot::vertexShader = std::string( R"(
#version 430 core

in float value;
uniform vec2  xScale;
uniform vec2  yScale;

void main()
{
    gl_Position = vec4(fma(xScale.x, gl_VertexID, xScale.y), 
                       fma(yScale.x, value,       yScale.y),
                       0.0f, 1.0f);
}
)");

/**
 * Simple GLSL Fragment shader. Simply outputs the color passed as parameter.
 */
const std::string SimplePlot::fragmentShader = std::string(R"(
#version 430 core

uniform vec4 color;
out vec4 outColor;

void main()
{
    outColor = color;
}
)");

SimplePlot::SimplePlot(const GLContext::Ptr& context) :
    Renderer(context, vertexShader, fragmentShader),
    dataRange_(0,0),
    color_({0.0, 0.8, 1.0, 1.0})
{}

SimplePlot::Ptr SimplePlot::Create(const GLContext::Ptr& context)
{
    return Ptr(new SimplePlot(context));
}

void SimplePlot::update_range()
{
    dataRange_.lower = reductor_.min(data_);
    dataRange_.upper = reductor_.max(data_);
}

void SimplePlot::draw(const View::ConstPtr& view) const
{
    if(data_.size() == 0) {
        return;
    }

    glUseProgram(this->renderProgram_);

    data_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glUniform2f(glGetUniformLocation(renderProgram_, "xScale"), 2.0f / data_.size(), -1.0f);
    glUniform2f(glGetUniformLocation(renderProgram_, "yScale"),
                2.0f / (1.1*dataRange_.length()), 0.0f);

    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
                 1, (const float*)&color_);

    glDrawArrays(GL_LINE_STRIP, 0, data_.size());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

} //namespace display
} //namespace rtac
