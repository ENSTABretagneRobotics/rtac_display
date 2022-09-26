#include <rtac_display/GLFrameBuffer.h>

namespace rtac { namespace display {

GLFrameBuffer::GLFrameBuffer() :
    glId_(0)
{
    glGenFramebuffers(1, &glId_);
}

GLFrameBuffer::~GLFrameBuffer()
{
    if(glId_)
        glDeleteFramebuffers(1, &glId_);
}

GLFrameBuffer::Ptr GLFrameBuffer::Create()
{
    return Ptr(new GLFrameBuffer());
}

void GLFrameBuffer::bind(GLenum target) const
{
    glBindFramebuffer(target, this->gl_id());
}

bool GLFrameBuffer::is_complete(GLenum target) const
{
    this->bind(target);
    bool res = glCheckFramebufferStatus(target) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(target, 0);

    return res;
}

} //namespace display
} //namespace rtac


