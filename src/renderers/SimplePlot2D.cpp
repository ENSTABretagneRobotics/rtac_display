#include <rtac_display/renderers/SimplePlot2D.h>

#include <algorithm>

namespace rtac { namespace display {

const std::string SimplePlot2D::vertexShader_xFromData = std::string( R"(
#version 430 core

in float x;
in float y;
uniform mat4 view;

void main()
{
    gl_Position = view*vec4(x, y, 0.0, 1.0);
}
)");

const std::string SimplePlot2D::vertexShader_xFromIndexes = std::string( R"(
#version 430 core

in float y;
uniform mat4 view;

void main()
{
    gl_Position = view*vec4(gl_VertexID, y, 0.0, 1.0);
}
)");

/**
 * Simple GLSL Fragment shader. Simply outputs the color passed as parameter.
 */
const std::string SimplePlot2D::fragmentShader = std::string(R"(
#version 430 core

uniform vec4 color;
out vec4 outColor;

void main()
{
    outColor = color;
}
)");

SimplePlot2D::SimplePlot2D(const GLContext::Ptr& context, const PlotData2D::Ptr& data) :
    Renderer(context, vertexShader_xFromData, fragmentShader),
    data_(data),
    color_({0.0, 0.8, 1.0, 1.0}),
    xFromData_(this->renderProgram_),
    xFromIndexes_(create_render_program(vertexShader_xFromIndexes, fragmentShader)),
    drawMode_(GL_POINTS)
{}

SimplePlot2D::Ptr SimplePlot2D::Create(const GLContext::Ptr& context, const PlotData2D::Ptr& data)
{
    if(!data) {
        return Ptr(new SimplePlot2D(context, PlotData2D::Create()));
    }
    else {
        return Ptr(new SimplePlot2D(context, data));
    }
}

void SimplePlot2D::draw(const View::ConstPtr& view) const
{
    if(data_->size() == 0) {
        return;
    }

    if(data_->x().size() > 0) {
        this->draw_from_data(view);
    }
    else {
        this->draw_from_indexes(view);
    }
}

void SimplePlot2D::draw_from_data(const View::ConstPtr& view) const
{
    glUseProgram(this->xFromData_);

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

    glDrawArrays(drawMode_, 0, data_->size());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

void SimplePlot2D::draw_from_indexes(const View::ConstPtr& view) const
{
    glUseProgram(this->xFromIndexes_);

    data_->y().bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    View::Mat4 viewMatrix = view->view_matrix();
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
                 1, (const float*)&color_);

    glDrawArrays(drawMode_, 0, data_->size());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

} //namespace display
} //namespace rtac
