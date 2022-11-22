#ifndef _DEF_RTAC_DISPLAY_FAN_RENDERER_H_
#define _DEF_RTAC_DISPLAY_FAN_RENDERER_H_

#include <memory>
#include <iostream>
#include <cmath>

#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Rectangle.h>
#include <rtac_base/interpolation.h>
#include <rtac_base/types/SonarPing2D.h>

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

    using Ptr      = std::shared_ptr<FanRenderer>;
    using ConstPtr = std::shared_ptr<const FanRenderer>;

    using Shape     = View::Shape;
    using Mat4      = View::Mat4;
    using Point2    = rtac::Point2<float>;
    using Point4    = rtac::Point4<float>;
    using Interval  = rtac::Bounds<float>;
    using Rectangle = rtac::Rectangle<float>;

    using Interpolator = rtac::algorithm::Interpolator<float>;

    static const std::string& vertexShader;
    static const std::string& fragmentShader;
    static const std::string& fragmentShaderNonLinear;

    enum class Direction : uint8_t {
        Left  = 0,
        Right = 1,
        Up    = 2,
        Down  = 3,
    };

    protected:

    GLTexture::Ptr data_;
    Colormap::Ptr  colormap_;
    Interval       valueRange_;
    GLReductor     reductor_;

    Interval         angle_;
    Interval         range_;
    Rectangle        bounds_;
    GLVector<Point4> corners_;
    Direction        direction_;

    GLTexture::Ptr bearingMap_;
    GLuint         linearBearingsProgram_;
    GLuint         nonlinearBearingsProgram_;

    void compute_scale(const GLVector<float>& data);

    FanRenderer(const GLContext::Ptr& context);

    public:

    static Ptr Create(const GLContext::Ptr& context);

    void set_value_range(Interval valueRange);

    void set_geometry_degrees(const Interval& angle, const Interval& range);
    void set_geometry(Interval angle, const Interval& range);
    void set_aperture(Interval angle);
    void set_range(Interval range);
    void set_direction(Direction dir) { direction_ = dir; }

    void set_data(const GLTexture::Ptr& tex);
    void set_data(const Shape& shape, const float* data);
    void set_data(const Shape& shape, const GLVector<float>& data,
                  bool computeScale = true);

    void set_bearings(unsigned int nBeams, const float* bearings,
                      unsigned int mapSize = 0);
    void enable_bearing_map();
    void disable_bearing_map();

    virtual void draw(const View::ConstPtr& view) const;

    Mat4 compute_view(const Shape& screen) const;

    GLTexture::Ptr      texture()       { return data_; }
    GLTexture::ConstPtr texture() const { return data_; }

    template <typename T, template<typename>class VectorT>
    void set_ping(const rtac::SonarPing2D<T,VectorT>& ping);
};

template <typename T, template<typename>class VectorT>
void FanRenderer::set_ping(const rtac::SonarPing2D<T,VectorT>& ping)
{
    this->set_bearings(ping.bearing_count(), ping.bearings().data());
    this->set_range(ping.range_bounds());

    data_->set_image({ping.width(), ping.height()}, ping.ping_data().data());
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_FAN_RENDERER_H_
