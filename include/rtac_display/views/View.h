#ifndef _DEF_RTAC_DISPLAY_VIEW_H_
#define _DEF_RTAC_DISPLAY_VIEW_H_

#include <iostream>
#include <memory>

#include <rtac_base/types/common.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Point.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

/**
 * Generic class for handling view geometry. This includes the position of the
 * camera in the 2D or 3D scene, the projection on the screen (2D, 3D
 * perspective...) and the aspect ratio (compensates for a resize of the
 * Display surface).
 *
 * All Renderer instances must be assigned an instance of View or of one of its
 * subtypes.  A View instance can be either a simple resizing to fit a 2D image
 * on the screen while keeping a constant aspect ratio, or a more complex 3D
 * perspective projection to render a 3D image.
 *
 * A 3D view projection can be fully described by a single 4x4 Homogeneous
 * matrix which handle both the projection on the screen and the point of view
 * in a 3D scene. In this implementation, the projection matrix and the point
 * of view matrice are kept separated and it is the job of the vertex shader to
 * combine them. (This allows the vertex shader to make custom 3D operations on
 * the vertices before projecting them on the 2D screen space).
 *
 * It is also the job of the view object to compensate for the variations of
 * the size of the display area (i.e. when the Display window is resized by the
 * user). The View::update_projection method must be reimplemented in
 * subclasses to implement this feature.
 *
 * Note : none of the View class and its derivatives make any call to OpenGL
 * API.  The matrices are fully handled from host-side by Eigen::Matrix4<float>
 * matrices. The Eigen matrices data are passed to OpenGL at each render by the
 * Renderer object. Both OpenGL and Eigen use column-major matrices by default
 * which makes the copy from Eigen to OpenGL seamless. 
 *
 * See [here](https://learnopengl.com/Getting-started/Coordinate-Systems) for
 * more information on OpenGL coordinate system.
 */
class View
{
    public:

    using Ptr      = std::shared_ptr<View>;
    using ConstPtr = std::shared_ptr<const View>;

    using Mat4   = rtac::Matrix4<float>;
    using Shape  = rtac::display::Shape;
    using Point2 = rtac::Point2<float>;
    using Point3 = rtac::Point3<float>;

    protected:

    Shape screenSize_;
    Mat4  projectionMatrix_;

    virtual void update_projection();

    public:

    static Ptr New(const Mat4& mat = Mat4::Identity());
    static Ptr Create(const Mat4& mat = Mat4::Identity());

    View(const Mat4& mat = Mat4::Identity());
    
    void set_screen_size(const Shape& screen);

    Mat4 projection_matrix() const;
    virtual Mat4 view_matrix() const;

    Shape screen_size() const;

    // below here are math helpers to create some projection matrices.
    public:

    static Mat4 from_corners(const Point2& lowerleft, const Point2& topRight);
};

inline View::Mat4 View::from_corners(const Point2& lowerLeft, const Point2& topRight)
{
    Mat4 view = Mat4::Identity();

    view(0,0) = 2.0f / (topRight.x - lowerLeft.x);
    view(0,3) = -0.5f*view(0,0) * (lowerLeft.x + topRight.x);
    view(1,1) = 2.0f / (topRight.y - lowerLeft.y);
    view(1,3) = -0.5f*view(1,1) * (lowerLeft.y + topRight.y);

    return view;
}

// inline View::Mat4 View::from_corners(const Point3& lowerleft, const Point3& topRight)
// {
//     Mat4 view = Mat4::Identity();
// 
//     view(0,0) = 2.0f / (topRight.x - lowerLeft.x);
//     view(0,3) = -0.5f*view(0,0) * (lowerLeft.x + topRight.x);
//     view(1,1) = 2.0f / (topRight.y - lowerLeft.y);
//     view(1,3) = -0.5f*view(1,1) * (lowerLeft.y + topRight.y);
//     view(2,2) = 2.0f / (topRight.z - lowerLeft.z);
//     view(2,3) = -0.5f*view(1,1) * (lowerLeft.z + topRight.z);
// 
//     return view;
// }

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_VIEW_H_
