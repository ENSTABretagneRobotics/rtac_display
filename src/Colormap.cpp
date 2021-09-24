#include <rtac_display/Colormap.h>

namespace rtac { namespace display {

GLTexture& Colormap::texture()
{
    return texture_;
}

const GLTexture& Colormap::texture() const
{
    return texture_;
}

}; //namespace display
}; //namespace rtac

