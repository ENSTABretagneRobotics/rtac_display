#ifndef _DEF_RTAC_DISPLAY_GL_CONTEXT_H_
#define _DEF_RTAC_DISPLAY_GL_CONTEXT_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/GLState.h>

namespace rtac { namespace display {

/**
 * Abstract class representing a GLContext.
 *
 * To be reimplemented for each window manager (GLFW, Qt...)
 */
class GLContext : public GLState
{
    public:

    using Ptr      = rtac::types::Handle<GLContext>;
    using ConstPtr = rtac::types::Handle<const GLContext>;

    protected:

    GLContext() {}

    public:

    virtual ~GLContext() = default;

    //virtual void make_current() const = 0;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_CONTEXT_H_
