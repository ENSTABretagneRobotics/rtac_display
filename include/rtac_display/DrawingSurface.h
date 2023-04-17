#ifndef _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_
#define _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_

#include <memory>
#include <utility>

#include <rtac_base/types/Point.h>

#include <rtac_display/GLContext.h>
#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/text/TextRenderer.h>

namespace rtac { namespace display {

/**
 * A generic plotting surface containing elements to draw.
 *
 * This object contains Renderer objects and successively draw them when the
 * DrawingSurface::draw method is called. It also keeps track of its own size
 * and inform the renderers of size changes. Being itself a Renderer, it can be
 * nested under other DrawingSurface instances.
 *
 * This is the base of the Display class which creates its own window.
 */
class DrawingSurface : public Renderer
{
    public :

    using Ptr      = std::shared_ptr<DrawingSurface>;
    using ConstPtr = std::shared_ptr<const DrawingSurface>;

    using Shape     = View::Shape;
    using Point2    = rtac::Point2<int>;
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::ConstPtr>;

    using RenderItem  = std::pair<Renderer::ConstPtr, View::Ptr>;
    using RenderItems = std::vector<RenderItem>;
    struct TextItem {
        text::TextRenderer::ConstPtr renderer;
        View::Ptr view;
        float anchorDepth;
        bool operator<(const TextItem& other) const { 
            return !(anchorDepth < other.anchorDepth);
        }
    };
    using TextItems   = std::vector<TextItem>;

    enum Flags : uint32_t {
        FLAGS_NONE  = 0x0,

        CLEAR_COLOR = 0x1,
        CLEAR_DEPTH = 0X2,
        
        GAMMA_CORRECTION = 0x10,
    };

    protected:
    
    Point2 viewportOrigin_;
    
    mutable RenderItems renderItems_;
    mutable TextItems   textItems_;

    Views        views_;
    Color::RGBAf clearColor_;
    Flags        displayFlags_;

    DrawingSurface(const GLContext::Ptr& context);

    public:

    static Ptr New(const GLContext::Ptr& context); // deprecated
    static Ptr Create(const GLContext::Ptr& context);

    void add_view(const View::Ptr& view);

    void add_render_item(const RenderItem& item);
    void add_render_item(const TextItem& item);
    void add_render_item(const Renderer::ConstPtr& renderer,
                         const View::Ptr& view);

    virtual void draw() { this->draw(View::New()); }
    virtual void draw(const View::ConstPtr& view) const override;

    void set_viewport_origin(const Point2& origin);
    void set_viewport_size(const Shape& size);
    void set_viewport(int x, int y, size_t width, size_t height);

    void set_clear_color(const Color::RGBAf& color);
    Color::RGBAf clear_color() const;

    void add_display_flags(Flags flags);
    void set_display_flags(Flags flags);
    void remove_display_flags(Flags flags);
    void handle_display_flags() const;

    template <class RendererT, class... Args>
    typename RendererT::Ptr create_renderer(const View::Ptr& view,
                                            const Args (&...args));
};

/**
 * Helper function to create a renderer of any type without having to specify
 * the GLContext or add it to the drawing surface.
 */
template <class RendererT, class... Args>
typename RendererT::Ptr DrawingSurface::create_renderer(const View::Ptr& view,
                                                        const Args &(...args))
{
    auto renderer =  RendererT::Create(this->context(), args...);
    this->add_render_item(renderer, view);
    return renderer;
}

}; //namespace display
}; //namespace rtac

inline rtac::display::DrawingSurface::Flags operator&(
    rtac::display::DrawingSurface::Flags lhs,
    rtac::display::DrawingSurface::Flags rhs)
{
    return static_cast<rtac::display::DrawingSurface::Flags>(
        static_cast<uint32_t>(lhs) &  static_cast<uint32_t>(rhs));
}

inline rtac::display::DrawingSurface::Flags& operator&=(
    rtac::display::DrawingSurface::Flags& lhs,
    rtac::display::DrawingSurface::Flags rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

inline rtac::display::DrawingSurface::Flags operator|(
    rtac::display::DrawingSurface::Flags lhs,
    rtac::display::DrawingSurface::Flags rhs)
{
    return static_cast<rtac::display::DrawingSurface::Flags>(
        static_cast<uint32_t>(lhs) |  static_cast<uint32_t>(rhs));
}

inline rtac::display::DrawingSurface::Flags& operator|=(
    rtac::display::DrawingSurface::Flags& lhs,
    rtac::display::DrawingSurface::Flags rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

#endif //_DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_
