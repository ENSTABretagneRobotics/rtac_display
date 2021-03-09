#include <rtac_display/samples/ImageDisplay.h>

namespace rtac { namespace display { namespace samples {

ImageDisplay::ImageDisplay(int width, int height, const std::string& title) :
    Display(width, height, title),
    renderer_(ImageRenderer::New())
{
    this->add_renderer(renderer_);
}

ImageRenderer::Ptr ImageDisplay::renderer()
{
    return renderer_;
}

ImageRenderer::ConstPtr ImageDisplay::renderer() const
{
    return renderer_;
}

}; //namespace samples
}; //namespace display
}; //namespace rtac

