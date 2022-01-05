#ifndef _DEF_RTAC_DISPLAY_RENDER_H_
#define _DEF_RTAC_DISPLAY_RENDER_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLContext.h>
#include <rtac_display/views/View.h>

namespace rtac { namespace display {

/**
 * Generic base class for rendering an object.
 *
 * This class implements the minimal interface to draw an object in an OpenGL
 * scene and is the base class for all other Renderer types. The Display object
 * takes a list of Renderer objects and at each render will call all of the
 * Renderer::draw() methods.
 *
 * To make a new type of Renderer, simply subclass the Renderer type and
 * override the Renderer::draw and optionally the Renderer::set_view methods.
 *
 * Without subclassing, the Renderer object will draw a X-Y-Z frame at the
 * origin.
 */
class Renderer
{
    public:

    using Ptr      = rtac::types::Handle<Renderer>;
    using ConstPtr = rtac::types::Handle<const Renderer>;

    using Shape = View::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    GLContext::Ptr    context_;
    GLuint            renderProgram_;
    mutable View::Ptr view_;

    public:

    static Ptr New(const std::string& vertexShader = vertexShader,
                   const std::string& fragmentShader = fragmentShader,
                   // Had to remove this because of eigen alignment issues. To be investigated
                   //const View::Ptr& view = View::New());
                   const View::Ptr& view = View::New());

    static Ptr New(const GLContext::Ptr& context,
                   const std::string& vertexShader = vertexShader,
                   const std::string& fragmentShader = fragmentShader,
                   // Had to remove this because of eigen alignment issues. To be investigated
                   //const View::Ptr& view = View::New());
                   const View::Ptr& view = View::New());

    Renderer(const GLContext::Ptr& context,
             const std::string& vertexShader = vertexShader,
             const std::string& fragmentShader = fragmentShader,
             // Had to remove this because of eigen alignment issues. To be investigated
             //const Viewi::Ptr& view = View::New());
             const View::Ptr& view = View::New());

    Renderer(const std::string& vertexShader = vertexShader,
             const std::string& fragmentShader = fragmentShader,
             // Had to remove this because of eigen alignment issues. To be investigated
             //const Viewi::Ptr& view = View::New());
             const View::Ptr& view = View::New());

    const GLContext::Ptr context() const { return context_; }
    
    virtual void draw();
    virtual void draw(View::ConstPtr view);
    virtual void set_view(const View::Ptr& view) const; // Why const ?

    View::Ptr view() const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDER_H_
