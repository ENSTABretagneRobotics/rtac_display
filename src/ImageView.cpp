#include <optix_helpers/display/ImageView.h>

namespace optix_helpers { namespace display {


ImageView::Ptr ImageView::New(const Shape& image)
{
    return Ptr(new ImageView(image));
}

ImageView::ImageView(const Shape& image) :
    image_(image)
{}

void ImageView::update_projection()
{
    projectionMatrix_ = Mat4::Identity();
    
    float metaRatio = screenSize_.ratio<float>() / image_.ratio<float>();
    if(metaRatio > 1.0f) {
        projectionMatrix_(0,0) = 1.0f / metaRatio;
    }
    else {
        projectionMatrix_(1,1) = metaRatio;
    }
}

void ImageView::set_image_shape(const Shape& image)
{
    image_ = image;
}

ImageView::Shape ImageView::image_shape() const
{
    return image_;
}

}; //namespace display
}; //namespace optix_helpers

