#ifndef _DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_

#include <iostream>

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>

namespace rtac { namespace display { namespace samples {

class ImageDisplay : public Display
{
    protected:

    ImageRenderer::Ptr renderer_;

    public:

    ImageDisplay(int width = 1024, int height = 768,
                 const std::string& title = "optix render");

    ImageRenderer::Ptr      renderer();
    ImageRenderer::ConstPtr renderer() const;

    template <typename T>
    void set_image(const Shape& imageSize, const T* data) {
        renderer_->set_image(imageSize, data);
    }
    template <typename T>
    void set_image(const Shape& imageSize, const GLVector<T>& data) {
        renderer_->set_image(imageSize, data);
    }
};

}; //namespace samples
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_
