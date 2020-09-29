#ifndef _DEF_OPTIX_HELPERS_DISPLAY_VIEW_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_VIEW_H_

#include <iostream>

#include <rtac_base/types/common.h>
#include <rtac_base/types/Shape.h>

#include <optix_helpers/Handle.h>

namespace optix_helpers { namespace display {

class ViewObj
{
    public:

    using Mat4  = rtac::types::Matrix4<float>;
    using Shape = rtac::types::Shape<size_t>;

    protected:

    Mat4 projectionMatrix_;

    public:

    ViewObj(const Mat4& mat = Mat4::Identity());
    
    virtual void update_projection(const Shape& screen);
    virtual Mat4 view_matrix(const Shape& screen);

    virtual Mat4 view_matrix() const;
};
using View = Handle<ViewObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_VIEW_H_
