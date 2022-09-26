#include <rtac_display/GLRenderBuffer.h>

namespace rtac { namespace display {

GLRenderBuffer::GLRenderBuffer(const Shape& shape, GLenum internalFormat) :
    glId_(0),
    shape_({0,0}),
    internalFormat_(0)
{
    glGenRenderbuffers(1, &glId_);
    this->resize(shape, internalFormat);
}

GLRenderBuffer::~GLRenderBuffer()
{
    glDeleteRenderbuffers(1, &glId_);
}

GLRenderBuffer::Ptr GLRenderBuffer::Create(const Shape& shape, GLenum internalFormat)
{
    return Ptr(new GLRenderBuffer(shape, internalFormat));
}

void GLRenderBuffer::resize(const Shape& shape, GLenum internalFormat)
{
    this->bind();
    glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, shape.width, shape.height);
    shape_          = shape;
    internalFormat_ = internalFormat;
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void GLRenderBuffer::bind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, glId_);
}

} //namespace display
} //namespace rtac
