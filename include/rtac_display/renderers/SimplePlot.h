#ifndef _DEF_RTAC_DISPLAY_RENDERER_SIMPLE_PLOT_H_
#define _DEF_RTAC_DISPLAY_RENDERER_SIMPLE_PLOT_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/GLVector.h>
#include <rtac_display/Color.h>
#include <rtac_display/GLReductor.h>
#include <rtac_display/renderers/Renderer.h>

namespace rtac { namespace display {

class SimplePlot : public Renderer
{
    public:

    using Ptr      = std::shared_ptr<SimplePlot>;
    using ConstPtr = std::shared_ptr<const SimplePlot>;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:

    GLVector<float> data_;
    Bounds<float>   dataRange_;
    Color::RGBAf    color_;

    GLReductor reductor_;

    public:

    SimplePlot(const GLContext::Ptr& context);

    public:

    static Ptr Create(const GLContext::Ptr& context);

    template <template<typename>class VectorT>
    void set_data(const VectorT<float>& data);
    void update_range();

    void set_color(const Color::RGBAf& color) { color_ = color; }
    void draw(const View::ConstPtr& view) const;
};

template <template<typename>class VectorT>
void SimplePlot::set_data(const VectorT<float>& data)
{
    data_ = data;
    this->update_range();
}

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_RENDERER_SIMPLE_PLOT_H_
