#ifndef _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_

#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/Handle.h>
#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/ImageView.h>

namespace optix_helpers { namespace display {

class ImageRenderer : public Renderer
{
    protected:

    void init_texture();

    public:

    using Ptr      = Handle<ImageRenderer>;
    using ConstPtr = Handle<const ImageRenderer>;

    using Mat4  = ImageView::Mat4;
    using Shape = ImageView::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:

    GLuint texId_;
    ImageView::Ptr imageView_;

    public:

    static Ptr New();

    ImageRenderer();
    ~ImageRenderer();
    
    void set_image(const Shape& imageSize, GLuint buffer);
    void set_image(const RenderBufferGL& buffer);
    
    virtual void draw();
};

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
