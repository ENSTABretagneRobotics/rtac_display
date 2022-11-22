#ifndef _DEF_RTAC_DISPLAY_IMAGE_VIEW_H_
#define _DEF_RTAC_DISPLAY_IMAGE_VIEW_H_

#include <iostream>
#include <memory>

#include <rtac_display/views/View.h>

namespace rtac { namespace display {

/**
 * Handle projection to display a simple 2D image on the screen. To be used
 * with the ImageRenderer object.
 *
 * This will manipulate the projection matrix to keep a constant aspect ratio
 * of 1 (image not stretched) regardless of the size of the Display area and
 * the size of the image.
 */
class ImageView : public View
{
    public:

    using Ptr      = std::shared_ptr<ImageView>;
    using ConstPtr = std::shared_ptr<const ImageView>;

    using Mat4  = View::Mat4;
    using Shape = View::Shape;

    protected:

    Shape image_;

    public:

    static Ptr New(const Shape& image = {1,1});
    static Ptr Create(const Shape& image = {1,1});

    ImageView(const Shape& image = {1,1});
    
    virtual void update_projection();
    void set_image_shape(const Shape& image);

    Shape image_shape() const;
};

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_IMAGE_VIEW_H_
