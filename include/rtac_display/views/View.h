#ifndef _DEF_RTAC_DISPLAY_VIEW_H_
#define _DEF_RTAC_DISPLAY_VIEW_H_

#include <iostream>

#include <rtac_base/types/common.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

class View
{
    public:
    
    // Alignment issue (caused by integration of pcl, activation of vectorization)
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr      = rtac::types::Handle<View>;
    using ConstPtr = rtac::types::Handle<const View>;

    using Mat4 = rtac::types::Matrix4<float>;
    using Shape = rtac::display::Shape;

    protected:

    Shape screenSize_;
    Mat4  projectionMatrix_;

    virtual void update_projection();

    public:

    static Ptr New(const Mat4& mat = Mat4::Identity());

    View(const Mat4& mat = Mat4::Identity());
    
    void set_screen_size(const Shape& screen);

    Mat4 projection_matrix() const;
    virtual Mat4 view_matrix() const;

    Shape screen_size() const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_VIEW_H_
