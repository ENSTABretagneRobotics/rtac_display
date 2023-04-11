#include <rtac_display/renderers/Scatter2D.h>

#include <algorithm>

namespace rtac { namespace display {

const std::string Scatter2D::vertexShader = std::string( R"(
#version 430 core

in float x;
in float y;
uniform mat4 view;

void main()
{
    gl_Position = view*vec4(x, y, 0.0, 1.0);
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

Scatter2D::Scatter2D(const GLContext::Ptr& context, const PlotData2D::Ptr& data) :
    Renderer(context, vertexShader, fragmentShader),
    data_(data),
    color_({0.0, 0.8, 1.0, 1.0}),
    dataMax_(-1.0)
{}

Scatter2D::Ptr Scatter2D::Create(const GLContext::Ptr& context, const PlotData2D::Ptr& data)
{
    if(!data) {
        return Ptr(new Scatter2D(context, PlotData2D::Create()));
    }
    else {
        return Ptr(new Scatter2D(context, data));
    }
}

void Scatter2D::draw(const View::ConstPtr& view) const
{
    if(data_->size() == 0) {
        return;
    }

    glPointSize(5);
    glUseProgram(this->renderProgram_);

    data_->x().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    data_->y().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    View::Mat4 viewMatrix = view->view_matrix();
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
                 1, (const float*)&color_);

    glDrawArrays(GL_POINTS, 0, data_->size());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

} //namespace display
} //namespace rtac
