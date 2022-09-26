#include <rtac_display/GLFrameBuffer.h>

namespace rtac { namespace display {

GLFrameBuffer::GLFrameBuffer() :
    glId_(0)
{
    glGenFrameBuffers(1, &glId_);
}

GLFrameBuffer::~GLFrameBuffer()
{
    if(glId_)
        glDeleteFrameBuffers(1, &glId_);
}

GLFrameBuffer::Ptr GLFrameBuffer::Create()
{
    return Ptr(new GLFrameBuffer());
}

void GLFrameBuffer::bind(GLenum target)
{
    glBindFrameBuffer(target, this->gl_id());
}

bool GLFrameBuffer::is_complete(GLenum target) const
{
    this->bind(target);
    bool res = glCheckFrameBufferStatus(target) == GL_FRAMEBUFFER_COMPLETE;
    glBindFrameBuffer(target, 0);

    return res;
}

} //namespace display
} //namespace rtac


