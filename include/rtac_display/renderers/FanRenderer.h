#ifndef _DEF_RTAC_DISPLAY_FAN_RENDERER_H_
#define _DEF_RTAC_DISPLAY_FAN_RENDERER_H_

#include <iostream>
#include <cmath>

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Rectangle.h>

#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/GLReductor.h>
#include <rtac_display/views/View.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/Colormap.h>
#include <rtac_display/colormaps/Viridis.h>

namespace rtac { namespace display {

class FanRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<FanRenderer>;
    using ConstPtr = rtac::types::Handle<const FanRenderer>;

    using Shape     = View::Shape;
    using Mat4      = View::Mat4;
    using Point2    = rtac::types::Point2<float>;
    using Point4    = rtac::types::Point4<float>;
    using Interval  = rtac::types::Interval<float>;
    using Rectangle = rtac::types::Rectangle<float>;

    static const std::string& vertexShader;
    static const std::string& fragmentShader;

    enum class Direction : uint8_t {
        Left   = 0,
        Right  = 1,
        Top    = 2,
        Bottom = 3,
    };


    protected:

    GLTexture::Ptr   data_;
    Interval         angle_;
    Interval         range_;
    Rectangle        bounds_;
    GLVector<Point4> corners_;
    Colormap::Ptr    colormap_;

    FanRenderer(const GLContext::Ptr& context);

    public:

    static Ptr Create(const GLContext::Ptr& context);

    void set_geometry_degrees(const Interval& angle, const Interval& range,
                              Direction dir = Direction::Top);
    void set_geometry(Interval angle, const Interval& range,
                      Direction dir = Direction::Top);

    virtual void draw(const View::ConstPtr& view) const;
    virtual void draw() const { std::cout << "There" << std::endl;}

    Mat4 compute_view(const Shape& screen) const;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_FAN_RENDERER_H_
