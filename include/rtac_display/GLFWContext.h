#ifndef _DEF_RTAC_DISPLAY_GLFW_CONTEXT_H_
#define _DEF_RTAC_DISPLAY_GLFW_CONTEXT_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/GLContext.h>

#include <GLFW/glfw3.h>

namespace rtac { namespace display {

class GLFWContext : public GLContext
{
    public:

    using Ptr      = rtac::Handle<GLFWContext>;
    using ConstPtr = rtac::Handle<const GLFWContext>;

    using Window = rtac::Handle<GLFWwindow>;
    
    protected:

    Window window_;

    GLFWContext(const Window& window) : 
        window_(window)
    {
        if(!window_) {
            throw std::runtime_error("Empty GLFW handle");
        }
    }

    public:

    static Ptr Create(const Window& window) {
        return Ptr(new GLFWContext(window));
    }

    //virtual void make_current() const {
    //    glfwMakeContextCurrent(window_.get());
    //}

    const Window& window() const { return window_; }
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GLFW_CONTEXT_H_
