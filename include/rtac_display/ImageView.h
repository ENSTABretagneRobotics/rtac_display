#ifndef _DEF_RTAC_DISPLAY_IMAGE_VIEW_H_
#define _DEF_RTAC_DISPLAY_IMAGE_VIEW_H_

#include <iostream>

#include <rtac_display/Handle.h>
#include <rtac_display/View.h>

namespace rtac { namespace display {

class ImageView : public View
{
    public:

    using Ptr      = Handle<ImageView>;
    using ConstPtr = Handle<const ImageView>;

    using Mat4  = View::Mat4;
    using Shape = View::Shape;

    protected:

    Shape image_;

    public:

    static Ptr New(const Shape& image = {1,1});

    ImageView(const Shape& image = {1,1});
    
    virtual void update_projection();
    void set_image_shape(const Shape& image);

    Shape image_shape() const;
};

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_IMAGE_VIEW_H_
