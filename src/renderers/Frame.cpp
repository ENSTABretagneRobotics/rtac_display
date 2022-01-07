#include <rtac_display/renderers/Frame.h>

namespace rtac { namespace display {

Frame::Ptr Frame::Create(const GLContext::Ptr& context,
                         const View3D::Pose& pose, const View::Ptr& view)
{
    return Ptr(new Frame(context, pose, view));
}

Frame::Frame(const GLContext::Ptr& context,
             const View3D::Pose& pose, const View::Ptr& view) :
    Renderer(context, Renderer::vertexShader, Renderer::fragmentShader, view),
    pose_(pose)
{}

Frame::Ptr Frame::New(const View3D::Pose& pose, const View::Ptr& view)
{
    return Ptr(new Frame(pose, view));
}

Frame::Frame(const View3D::Pose& pose, const View::Ptr& view) :
    Renderer(Renderer::vertexShader, Renderer::fragmentShader, view),
    pose_(pose)
{}

void Frame::set_pose(const View3D::Pose& pose)
{
    pose_ = pose;
}

void Frame::draw() const
{
    if(!this->view()) {
        throw std::runtime_error("No view in renderer");
    }
    this->draw(this->view());
}

void Frame::draw(const View::ConstPtr& view) const
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
    
    View3D::Mat4 viewMatrix = view->view_matrix()*pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());

    glDrawArrays(GL_LINES, 0, 6);
    
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);
    glLineWidth(lineWidth);
}

}; //namespace display
}; //namespace rtac

