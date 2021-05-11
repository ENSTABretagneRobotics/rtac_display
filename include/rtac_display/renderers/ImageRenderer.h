#ifndef _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/ImageView.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/GLTexture.h>

namespace rtac { namespace display {

class ImageRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<ImageRenderer>;
    using ConstPtr = rtac::types::Handle<const ImageRenderer>;

    using Mat4  = ImageView::Mat4;
    using Shape = ImageView::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    GLTexture::Ptr texture_;
    ImageView::Ptr imageView_;

    public:

    static Ptr New();

    ImageRenderer();

    GLTexture::Ptr& texture();
    GLTexture::ConstPtr texture() const;
    
    virtual void draw();
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
