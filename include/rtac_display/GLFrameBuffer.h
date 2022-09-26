#ifndef _DEF_RTAC_DISPLAY_GL_FRAMEBUFFER_H_
#define _DEF_RTAC_DISPLAY_GL_FRAMEBUFFER_H_

#include <iostream>
#include <memory>

#include <rtac_display/utils.h>

//#include <rtac_display/GLFrameBufferAttachment.h>

namespace rtac { namespace display {

class GLFrameBuffer
{
    public:

    using Ptr      = std::shared_ptr<GLFrameBuffer>;
    using ConstPtr = std::shared_ptr<const GLFrameBuffer>;

    protected:

    GLuint glId_;

    GLFrameBuffer();

    public:

    ~GLFrameBuffer();

    static Ptr Create();

    GLuint gl_id() const { return glId_; }

    void bind(GLenum target = GL_FRAMEBUFFER) const;

    bool is_complete(GLenum target = GL_FRAMEBUFFER) const;
};

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_FRAMEBUFFER_H_
