#include <optix_helpers/display/RenderBufferGL.h>

namespace optix_helpers { namespace display {

RenderBufferGLObj::RenderBufferGLObj(const Context& context, RTformat format,
                                     const std::string& name) :
    RenderBufferObj((*context)->createBufferFromGLBO(
                    RT_BUFFER_OUTPUT, RenderBufferGLObj::create_vbo()),
                    name),
    vboId_(this->buffer()->getGLBOId())
{
    this->buffer()->setFormat(format);
}

GLuint RenderBufferGLObj::create_vbo()
{
    GLuint vboId;
    glGenBuffers(1, &vboId);
    return vboId;
}

void RenderBufferGLObj::set_size(size_t width, size_t height)
{
    this->RenderBufferObj::set_size(width, height);
    size_t size = width*height*this->buffer()->getElementSize();
    glBindBuffer(GL_ARRAY_BUFFER, vboId_);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint RenderBufferGLObj::gl_id() const
{
    return vboId_;
}

//RenderBufferGL::RenderBufferGL()
//{}
//
//RenderBufferGL::RenderBufferGL(const Context& context, RTformat format,
//                               const std::string& name) :
//    Handle<RenderBufferGLObj>(context, format, name)
//{}
//
//RenderBufferGL::operator RenderBuffer()
//{
//    return RenderBuffer(std::dynamic_pointer_cast<RenderBufferObj>(this->obj_));
//}
//
//RenderBufferGL::operator Buffer()
//{
//    return Buffer(std::dynamic_pointer_cast<BufferObj>(this->obj_));
//}

}; //namespace display
}; //namespace optix_helpers

