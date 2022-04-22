#include <rtac_display/views/ImageView.h>

namespace rtac { namespace display {

/**
 * Creates a new ImageView instance on the heap.
 *
 * @param image the size of the image to be displayed.
 *
 * @return a shared pointer to the newly created ImageView instance.
 */
ImageView::Ptr ImageView::New(const Shape& image)
{
    return Ptr(new ImageView(image));
}

/**
 * Constructor of ImageView.
 *
 * @param image the size of the image to be displayed.
 */
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
    float screenRatio = screenSize_.ratio<float>();
    float imageRatio  = image_.ratio<float>();

    if(screenRatio > imageRatio) {
        // here the screen is wider than the image.
        projectionMatrix_(0,0) = 2.0f / (screenRatio * image_.height);
        projectionMatrix_(0,3) = -0.5f*image_.width * projectionMatrix_(0,0);
        projectionMatrix_(1,1) = 2.0f / image_.height;
        projectionMatrix_(1,3) = -1.0f;
    }
    else {
        // here the image is wider then the screen
        projectionMatrix_(0,0) = 2.0f / image_.width;
        projectionMatrix_(0,3) = -1.0f;
        projectionMatrix_(1,1) = 2.0f * screenRatio / image_.width;
        projectionMatrix_(1,3) = -0.5f*image_.height * projectionMatrix_(1,1);
    }
}

/**
 * Set width and height of image via a Shape object.
 *
 * @param image the size of the image to be displayed.
 */
void ImageView::set_image_shape(const Shape& image)
{
    image_ = image;
}

/**
 * @return the size of the image to be displayed.
 */
ImageView::Shape ImageView::image_shape() const
{
    return image_;
}

}; //namespace display
}; //namespace rtac

