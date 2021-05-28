#ifndef _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_
#define _DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_

#include <rtac_base/types/Handle.h>

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
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::Ptr>;

    protected:
    
    Views        views_;
    Renderers    renderers_;

    DrawingSurface(const Shape& shape);

    public:

    static Ptr New(const Shape& shape);

    void add_view(const View::Ptr& view);
    void add_renderer(const Renderer::Ptr& renderer);
    virtual void draw();
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_DRAWING_SURFACE_H_
