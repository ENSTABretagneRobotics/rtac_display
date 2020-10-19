#ifndef _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <optix_helpers/display/Handle.h>
#include <optix_helpers/display/utils.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class Renderer
{
    public:

    using Ptr      = Handle<Renderer>;
    using ConstPtr = Handle<const Renderer>;

    using Shape = View::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    GLuint  renderProgram_;
    mutable View::Ptr   view_;

    public:

    static Ptr New(const std::string& vertexShader = vertexShader,
                   const std::string& fragmentShader = fragmentShader,
                   // Had to remove this because of eigen alignment issues. To be investigated
                   //const View::Ptr& view = View::New());
                   const View::Ptr& view = View::New());

    Renderer(const std::string& vertexShader = vertexShader,
             const std::string& fragmentShader = fragmentShader,
             // Had to remove this because of eigen alignment issues. To be investigated
             //const Viewi::Ptr& view = View::New());
             const View::Ptr& view = View::New());
    
    virtual void draw();
    void set_view(const View::Ptr& view) const;

    View::Ptr view() const;
};

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
