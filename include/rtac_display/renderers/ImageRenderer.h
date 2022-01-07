#ifndef _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLContext.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/ImageView.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/Colormap.h>

#include <rtac_display/colormaps/Viridis.h>
#include <rtac_display/colormaps/Gray.h>

namespace rtac { namespace display {

/**
 * Simple Renderer to display an image.
 *
 * This takes a GLTexture as an image. See GLTexture documentation for more
 * information on how to handle images in OpenGL.
 *
 * For now, no colormaps are supported. Sending grayscale data will result in
 * displaying a red-scaled image. (OpenGL always displays a full RGBA image.
 * Missing blue and green components are filled with 0, and missing alpha is
 * filled with 1).
 */
class ImageRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<ImageRenderer>;
    using ConstPtr = rtac::types::Handle<const ImageRenderer>;

    using Mat4  = ImageView::Mat4;
    using Shape = ImageView::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;
    static const std::string colormapFragmentShader;

    protected:
    
    GLTexture::Ptr texture_;
    ImageView::Ptr imageView_;
    Colormap::Ptr  colormap_;

    GLuint passThroughProgram_;
    GLuint colormapProgram_;

    bool verticalFlip_;

    ImageRenderer(const GLContext::Ptr& context);
    ImageRenderer();

    public:

    static Ptr Create(const GLContext::Ptr& context);
    static Ptr New();

    GLTexture::Ptr& texture();
    GLTexture::ConstPtr texture() const;
    
    void draw(const GLTexture& texture);
    virtual void draw();

    void set_colormap(const Colormap::Ptr& colormap);
    bool enable_colormap();
    void disable_colormap();
    bool uses_colormap() const;
    void set_vertical_flip(bool doFlip);

    void set_viridis_colormap();
    void set_gray_colormap();
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
