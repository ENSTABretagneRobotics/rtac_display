#ifndef _DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_

#include <iostream>

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>

namespace rtac { namespace display { namespace samples {

class ImageDisplay : public Display
{
    public:

    using Shape = ImageRenderer::Shape;

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

template <typename T>
class DeviceImageDisplay : public ImageDisplay
{
    // This class contains a GLVector which is expected to be filled to update
    // the image. The main purpose of this class is to be able to easilly
    // display image generated on the device (generated with CUDA
    // for example).

    public:

    using Shape = ImageDisplay::Shape;
    
    protected:
    
    Shape       imageShape_;
    GLVector<T> data_;
    bool        imageUpdated_;
    
    public:

    DeviceImageDisplay(unsigned int width = 1024, unsigned int height = 768,
                       const std::string& title = "optix render");
    
    void resize_image(const Shape& shape);
    Shape image_shape() const;

    GLVector<T>& data();
    const GLVector<T>& data() const;

    virtual void draw();
};

template <typename T>
DeviceImageDisplay<T>::DeviceImageDisplay(unsigned int width, unsigned int height,
                                       const std::string& title) :
    ImageDisplay(width, height, title),
    imageShape_({width, height}),
    data_(width*height),
    imageUpdated_(false)
{}

template <typename T>
void DeviceImageDisplay<T>::resize_image(const Shape& shape)
{
    data_.resize(shape.area());
    imageShape_ = shape;
}

template <typename T>
typename DeviceImageDisplay<T>::Shape DeviceImageDisplay<T>::image_shape() const
{
    return imageShape_;
}

template <typename T>
GLVector<T>& DeviceImageDisplay<T>::data()
{
    imageUpdated_ = true;
    return data_;
}

template <typename T>
const GLVector<T>& DeviceImageDisplay<T>::data() const
{
    return data_;
}

template <typename T>
void DeviceImageDisplay<T>::draw()
{
    if(imageUpdated_) {
        renderer_->set_image(imageShape_, data_);
    }
    this->ImageDisplay::draw();
}

}; //namespace samples
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_IMAGE_DISPLAY_H_
