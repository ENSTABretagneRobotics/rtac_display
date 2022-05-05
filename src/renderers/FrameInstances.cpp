#include <rtac_display/renderers/FrameInstances.h>

namespace rtac { namespace display {

const std::string FrameInstances::vertexShader = std::string( R"(
#version 430 core

layout(location=0) in vec3 point;
layout(location=1) in vec3 color;
layout(location=2) in mat4 pose; // takes locations 2,3,4,5

uniform mat4 view;
out vec3 c;

void main()
{
    gl_Position = view*pose*vec4(point, 1.0f);
    c = color;
}
)");

const std::string FrameInstances::fragmentShader = std::string(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)");

FrameInstances::Ptr FrameInstances::Create(const GLContext::Ptr& context,
                         const View::Ptr& view, const View3D::Pose& pose)
{
    return Ptr(new FrameInstances(context, view, pose));
}

FrameInstances::FrameInstances(const GLContext::Ptr& context,
                               const View::Ptr& view,
                               const View3D::Pose& pose) :
    Renderer(context, vertexShader, fragmentShader, view),
    globalPose_(pose)
{}

FrameInstances::Ptr FrameInstances::New(const View::Ptr& view, const View3D::Pose& pose)
{
    return Ptr(new FrameInstances(view, pose));
}

FrameInstances::FrameInstances(const View::Ptr& view, const View3D::Pose& pose) :
    Renderer(vertexShader, fragmentShader, view),
    globalPose_(pose)
{}

void FrameInstances::draw() const
{
    if(!this->view()) {
        throw std::runtime_error("No view in renderer");
    }
    this->draw(this->view());
}

void FrameInstances::draw(const View::ConstPtr& view) const
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

    if(deviceData_.size() < poses_.size()) {
        deviceData_ = poses_;
    }

    GLfloat lineWidth;
    glGetFloatv(GL_LINE_WIDTH, &lineWidth);
    glLineWidth(3);

    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, colors);
    glEnableVertexAttribArray(1);

    deviceData_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16*sizeof(float),
                          (const void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 16*sizeof(float),
                          (const void*)(4*sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 16*sizeof(float),
                          (const void*)(8*sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 16*sizeof(float),
                          (const void*)(12*sizeof(float)));
    glEnableVertexAttribArray(5);

    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);

    View3D::Mat4 viewMatrix = view->view_matrix()*globalPose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());

    // glDrawArrays(GL_LINES, 0, 6);
    glDrawArraysInstanced(GL_LINES, 0, 6, deviceData_.size());
    
    deviceData_.unbind(GL_ARRAY_BUFFER);

    glDisableVertexAttribArray(5);
    glDisableVertexAttribArray(4);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);
    glLineWidth(lineWidth);
}

}; //namespace display
}; //namespace rtac

