#ifndef _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_

#include <iostream>

#include <optix_helpers/Handle.h>

#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class ImageViewObj : public ViewObj
{
    public:

    using Mat4  = ViewObj::Mat4;
    using Shape = ViewObj::Shape;

    protected:

    Shape image_;

    public:

    ImageViewObj(const Shape& image = {1,1});
    
    virtual void update_projection(const Shape& screen);
    void set_image_shape(const Shape& image);
};

using ImageView = Handle<ImageViewObj>;
//class ImageView : public Handle<ImageViewObj>
//{
//    public:
//
//    using Mat4  = ImageViewObj::Mat4;
//    using Shape = ImageViewObj::Shape;
//
//    ImageView(const Shape& image = {1,1});
//
//    operator View();
//};

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_IMAGE_VIEW_H_
