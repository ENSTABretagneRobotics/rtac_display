#ifndef _RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_
#define _RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/GLVector.h>
#include <rtac_display/Color.h>
#include <rtac_display/GLReductor.h>
#include <rtac_display/renderers/Renderer.h>

namespace rtac { namespace display {

class Scatter2D : public Renderer
{
    public:

    using Ptr      = std::shared_ptr<Scatter2D>;
    using ConstPtr = std::shared_ptr<const Scatter2D>;

    static const std::string vertexShader;
    static const std::string fragmentShader;
    
    protected:

    GLVector<float> x_;
    GLVector<float> y_;
    Bounds<float>   xRange_;
    Bounds<float>   yRange_;
    Color::RGBAf    color_;

    GLReductor reductor_;

    Scatter2D(const GLContext::Ptr& context);

    public:

    static Ptr Create(const GLContext::Ptr& context);
    
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& x, const VectorT<float>& y);
    void update_range();

    void set_color(const Color::RGBAf& color) { color_ = color; }
    void draw(const View::ConstPtr& view) const;
};

template <template<typename>class VectorT>
bool Scatter2D::set_data(const VectorT<float>& x, const VectorT<float>& y)
{
    if(x.size() != y.size()) {
        return false;
    }
    x_ = x;
    y_ = y;
    auto ptr = y_.map();
    this->update_range();
    return true;
}

} //namespace display
} //namespace rtac

#endif //_RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_
