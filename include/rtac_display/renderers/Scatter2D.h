#ifndef _RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_
#define _RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/PlotData2D.h>
#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/PlotView2D.h>

namespace rtac { namespace display {

class Scatter2D : public Renderer
{
    public:

    using Ptr      = std::shared_ptr<Scatter2D>;
    using ConstPtr = std::shared_ptr<const Scatter2D>;

    static const std::string vertexShader;
    static const std::string fragmentShader;
    
    protected:

    PlotData2D::Ptr data_;
    Color::RGBAf    color_;

    Scatter2D(const GLContext::Ptr& context, const PlotData2D::Ptr& data);

    public:

    float dataMax_;

    static Ptr Create(const GLContext::Ptr& context,
                      const PlotData2D::Ptr& data = nullptr);
    
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& x, const VectorT<float>& y);
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& y);

    void set_color(const Color::RGBAf& color) { color_ = color; }
    void draw(const View::ConstPtr& view) const override;
};

template <template<typename>class VectorT>
bool Scatter2D::set_data(const VectorT<float>& x, const VectorT<float>& y)
{
    return data_->set_data(x,y);
}

template <template<typename>class VectorT>
bool Scatter2D::set_data(const VectorT<float>& y)
{
    return data_->set_data(y);
}

} //namespace display
} //namespace rtac

#endif //_RTAC_DISPLAY_RENDERERS_SCATTER_2D_H_
