#ifndef _DEF_RTAC_DISPLAY_GL_RENDER_BUFFER_H_
#define _DEF_RTAC_DISPLAY_GL_RENDER_BUFFER_H_

#include <memory>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

class GLRenderBuffer
{
    public:

    using Ptr      = std::shared_ptr<GLRenderBuffer>;
    using ConstPtr = std::shared_ptr<const GLRenderBuffer>;

    using Shape = rtac::display::Shape;

    protected:

    GLuint glId_;
    Shape  shape_;
    GLenum internalFormat_;

    GLRenderBuffer(const Shape& shape, GLenum internalFormat);

    public:

    ~GLRenderBuffer();

    static Ptr Create(const Shape& shape = Shape{0,0},
                      GLenum internalFormat = 0);

    void resize(const Shape& shape) { this->resize(shape, internalFormat_); }
    void resize(const Shape& shape, GLenum internalFormat);

    void bind() const;

    GLuint gl_id()           const { return glId_;  }
    Shape  shape()           const { return shape_; }
    GLenum internal_format() const { return internalFormat_; }
};

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_RENDER_BUFFER_H_
