#ifndef _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_

#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/ImageView.h>

namespace optix_helpers { namespace display {

class ImageRendererObj : public RendererObj
{
    public:

    using Mat4  = ImageViewObj::Mat4;
    using Shape = ImageViewObj::Shape;

    ImageRendererObj();
};

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_IMAGE_RENDERER_H_
