#ifndef _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_

#include <iostream>

#include <optix_helpers/Handle.h>

#include <optix_helpers/display/Handle.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

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
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_
