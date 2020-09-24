#ifndef _DEF_OPTIX_HELPERS_DISPLAY_RENDER_BUFFER_GL_
#define _DEF_OPTIX_HELPERS_DISPLAY_RENDER_BUFFER_GL_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

namespace optix_helpers { namespace display {

class RenderBufferGLObj : public RenderBufferObj
{
    protected:
    
    GLuint vboId_;

    static GLuint create_vbo();

    public:

    RenderBufferGLObj(const Context& context, RTformat format,
                      const std::string& name);

    virtual void set_size(size_t width, size_t height);
    GLuint gl_id();
};

class RenderBufferGL : public Handle<RenderBufferGLObj>
{
    public:
    
    RenderBufferGL();
    RenderBufferGL(const Context& context, RTformat format,
                   const std::string& name);

    operator RenderBuffer();
    operator Buffer();
};

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_RENDER_BUFFER_GL_
