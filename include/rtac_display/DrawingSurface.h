#ifndef _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_
#define _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Point.h>

#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>

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

    using Ptr      = rtac::types::Handle<DrawingSurface>;
    using ConstPtr = rtac::types::Handle<const DrawingSurface>;

    using Shape     = View::Shape;
    using Point2    = types::Point2<int>;
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::Ptr>;

    enum Flags : uint32_t {
        FLAGS_NONE  = 0x0,

        CLEAR_COLOR = 0x1,
        CLEAR_DEPTH = 0X2,
        
        GAMMA_CORRECTION = 0x10,
    };

    protected:
    
    Point2       viewportOrigin_;
    Views        views_;
    Renderers    renderers_;
    Color::RGBAf clearColor_;
    Flags        displayFlags_;

    DrawingSurface(const Shape& shape);

    public:

    static Ptr New(const Shape& shape);

    void add_view(const View::Ptr& view);
    void add_renderer(const Renderer::Ptr& renderer);
    virtual void draw();

    void set_viewport_origin(const Point2& origin);
    void set_viewport_size(const Shape& size);
    void set_viewport(int x, int y, size_t width, size_t height);

    void add_display_flags(Flags flags);
    void set_display_flags(Flags flags);
    void remove_display_flags(Flags flags);
    void handle_display_flags() const;
};

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
