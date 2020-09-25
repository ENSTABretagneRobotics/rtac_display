#include <optix_helpers/display/ImageView.h>

namespace optix_helpers { namespace display {


ImageViewObj::ImageViewObj(const Shape& image) :
    image_(image)
{}

void ImageViewObj::update_projection(const Shape& screen)
{
    projectionMatrix_ = Mat4::Identity();
    
    float metaRatio = screen.ratio<float>() / image_.ratio<float>();
    if(metaRatio > 1.0f) {
        projectionMatrix_(0,0) = 1.0f / metaRatio;
    }
    else {
        projectionMatrix_(1,1) = metaRatio;
    }
}

void ImageViewObj::set_image_shape(const Shape& image)
{
    image_ = image;
}
ImageView::ImageView(const Shape& image) :
    Handle<ImageViewObj>(image)
{}

ImageView::operator View()
{
    return View(std::dynamic_pointer_cast<ViewObj>(this->obj_));
}

}; //namespace display
}; //namespace optix_helpers

