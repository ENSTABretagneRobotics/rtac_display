#ifndef _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_

#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/ImageView.h>

namespace optix_helpers { namespace display {

class ImageRendererObj : public RendererObj
{
    protected:

    void init_texture();

    public:

    using Mat4  = ImageViewObj::Mat4;
    using Shape = ImageViewObj::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    GLuint texId_;
    ImageView imageView_;

    ImageRendererObj();
    ~ImageRendererObj();
    
    void set_image(const Shape& imageSize, GLuint buffer);
    void set_image(const RenderBufferGL& buffer);
    
    virtual void draw();
};
using ImageRenderer = Handle<ImageRendererObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
