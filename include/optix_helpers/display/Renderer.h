#ifndef _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

#include <optix_helpers/display/utils.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class RendererObj
{
    protected:
    
    GLuint renderProgram_;
    View   view_;

    public:

    RendererObj(const Source& vertexShader, const Source& fragmentShader,
                const View& view);

    virtual void draw();
};

class Renderer : Handle<RendererObj>
{
    public:

    Renderer();
    Renderer(const Source& vertexShader, const Source& fragmentShader,
             const View& view);
    Renderer(const std::shared_ptr<RendererObj>& obj);
};

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
