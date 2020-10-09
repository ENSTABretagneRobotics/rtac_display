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
    
    // Alignment issue (caused by integration of pcl, activation of vectorization)
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Mat4  = rtac::types::Matrix4<float>;
    using Shape = rtac::types::Shape<size_t>;

    protected:

    Shape screenSize_;
    Mat4  projectionMatrix_;

    virtual void update_projection();

    public:

    ViewObj(const Mat4& mat = Mat4::Identity());
    
    void set_screen_size(const Shape& screen);

    Mat4 projection_matrix() const;
    virtual Mat4 view_matrix() const;
};
using View = Handle<ViewObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_VIEW_H_
