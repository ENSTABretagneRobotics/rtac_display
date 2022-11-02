#include <rtac_display/samples/ImageDisplay.h>

namespace rtac { namespace display { namespace samples {

ImageDisplay::ImageDisplay(int width, int height, const std::string& title) :
    Display(width, height, title),
    renderer_(this->create_renderer<ImageRenderer>(View::Create()))
{}

}; //namespace samples
}; //namespace display
}; //namespace rtac

