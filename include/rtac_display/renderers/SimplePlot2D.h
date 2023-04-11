#ifndef _RTAC_DISPLAY_RENDERERS_SIMPLE_PLOT_2D_H_
#define _RTAC_DISPLAY_RENDERERS_SIMPLE_PLOT_2D_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/PlotData2D.h>
#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/PlotView2D.h>

namespace rtac { namespace display {

class SimplePlot2D : public Renderer
{
    public:

    using Ptr      = std::shared_ptr<SimplePlot2D>;
    using ConstPtr = std::shared_ptr<const SimplePlot2D>;

    static const std::string vertexShader_xFromData;
    static const std::string vertexShader_xFromIndexes;
    static const std::string fragmentShader;
    
    protected:

    PlotData2D::Ptr data_;
    Color::RGBAf    color_;

    GLuint xFromData_;
    GLuint xFromIndexes_;

    GLenum drawMode_;

    SimplePlot2D(const GLContext::Ptr& context, const PlotData2D::Ptr& data);

    void draw_from_data(const View::ConstPtr& view) const;
    void draw_from_indexes(const View::ConstPtr& view) const;

    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const PlotData2D::Ptr& data = nullptr);
    
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& x, const VectorT<float>& y);
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& y);

    void set_color(const Color::RGBAf& color) { color_ = color; }
    void set_draw_mode(GLenum drawMode) { drawMode_ = drawMode; }
    void draw(const View::ConstPtr& view) const override;
};

template <template<typename>class VectorT>
bool SimplePlot2D::set_data(const VectorT<float>& x, const VectorT<float>& y)
{
    return data_->set_data(x,y);
}

template <template<typename>class VectorT>
bool SimplePlot2D::set_data(const VectorT<float>& y)
{
    return data_->set_data(y);
}

} //namespace display
} //namespace rtac

#endif //_RTAC_DISPLAY_RENDERERS_SIMPLE_PLOT_2D_H_
