#include <rtac_display/views/ImageView.h>

namespace rtac { namespace display {


ImageView::Ptr ImageView::New(const Shape& image)
{
    return Ptr(new ImageView(image));
}

ImageView::ImageView(const Shape& image) :
    image_(image)
{}

/**
 * This updates the projection matrix by taking into account the size of the
 * window and the size of the image.
 *
 * In the ImageRenderer class, the coordinates of the corners of the image are
 * fixed to [left,right] = [-1,1], [bottom,top] = [-1,1]. The ImageView class
 * will manipulate the projection matrix to display the biggest image possible
 * while keeping the apparent aspect ratio to 1.
 *
 * See [here](https://learnopengl.com/Getting-started/Coordinate-Systems) for
 * more information on OpenGL coordinate system.
 */
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

/**
 * Set width and height of image via a Shape object.
 */
void ImageView::set_image_shape(const Shape& image)
{
    image_ = image;
}

ImageView::Shape ImageView::image_shape() const
{
    return image_;
}

}; //namespace display
}; //namespace rtac

