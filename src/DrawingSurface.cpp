#include <rtac_display/DrawingSurface.h>

namespace rtac { namespace display {

DrawingSurface::DrawingSurface(const GLContext::Ptr& context, const Shape& shape) :
    Renderer(context, "", ""),
    viewportOrigin_({0,0}),
    clearColor_({0,0,0,0}),
    displayFlags_(FLAGS_NONE)
{}

/**
 * Instanciate a new DrawingSurface.
 *
 * @param shape initial {width,height} of the DrawingSurface.
 *
 * @return a shared pointer to the newly instanciated DrawingSurface.
 */
DrawingSurface::Ptr DrawingSurface::New(const GLContext::Ptr& context, const Shape& shape)
{
    return Ptr(new DrawingSurface(context, shape));
}

/**
 * Append a rtac::display::View to the list of handled views.
 *
 * This allows DrawingSurface to inform the views of a change of size of the
 * display area (i.e.  when the window is resized). This allows the views to
 * compensate for a change in the aspect ratio in the display area.
 *
 * If the view is already handled, it won't be added a second time.
 */
void DrawingSurface::add_view(const View::Ptr& view)
{
    for(auto v : views_) {
        if(view.get() == view.get()) {
            return;
        }
    }
    views_.push_back(view);
}

void DrawingSurface::add_render_item(const RenderItem& item)
{
    renderItems_.push_back(item);
    this->add_view(item.second);
}

void DrawingSurface::add_render_item(const TextItem& item)
{
    textItems_.push_back(item);
    this->add_view(item.view);
}

void DrawingSurface::add_render_item(const Renderer::ConstPtr& renderer,
                                     const View::Ptr& view)
{
    if(!renderer || ! view) {
        return;
    }
    auto text = std::dynamic_pointer_cast<const text::TextRenderer>(renderer);
    if(text) {
        this->add_render_item(TextItem({text, view, 0.0f}));
    }
    else {
        this->add_render_item(std::make_pair(renderer, view));
    }
}

/**
 * Update all handled views with the current display size and draw all the
 * handled renderers after clearing the display area.
 */
void DrawingSurface::draw(const View::ConstPtr& view)
{
    Shape shape = view->screen_size();
    for(auto view : views_) {
        view->set_screen_size(shape);
    }
    
    glViewport(viewportOrigin_.x, viewportOrigin_.y,
               shape.width, shape.height);

    this->handle_display_flags();
    for(auto& item : renderItems_) {
        if(item.first) {
            item.first->draw(item.second);
        }
    }
    
    // Transparency rendering (as text background) must be renderered closest
    // last (hence sorting text items with respect to anchor_depth).
    for(auto& item : textItems_) {
        if(item.renderer) {
            item.anchorDepth = item.renderer->anchor_depth(item.view);
        }
    }
    std::sort(textItems_.begin(), textItems_.end());
    for(auto& item : textItems_) {
        if(item.renderer) {
            item.renderer->draw(item.view);
        }
    }
    glDisable(GL_FRAMEBUFFER_SRGB);
}

void DrawingSurface::set_viewport_origin(const Point2& origin)
{
    viewportOrigin_ = origin;
}

void DrawingSurface::set_viewport_size(const Shape& size)
{}

void DrawingSurface::set_viewport(int x, int y, size_t width, size_t height)
{
    this->set_viewport_origin({x,y});
    this->set_viewport_size({width,height});
}

void DrawingSurface::set_clear_color(const Color::RGBAf& color)
{
    clearColor_ = color;
}

Color::RGBAf DrawingSurface::clear_color() const
{
    return clearColor_;
}

void DrawingSurface::add_display_flags(Flags flags)
{
    displayFlags_ |= flags;
}

void DrawingSurface::set_display_flags(Flags flags)
{
    displayFlags_ = flags;
}

void DrawingSurface::remove_display_flags(Flags flags)
{
    displayFlags_ &= static_cast<Flags>(~flags);
}

void DrawingSurface::handle_display_flags() const
{
    GLbitfield clearingMask = 0;
    if(displayFlags_ & CLEAR_COLOR) {
        clearingMask |= GL_COLOR_BUFFER_BIT;
        glClearColor(clearColor_.r,
                     clearColor_.g,
                     clearColor_.b,
                     clearColor_.a);
    }
    if(displayFlags_ & CLEAR_DEPTH) clearingMask |= GL_DEPTH_BUFFER_BIT;

    if(displayFlags_ & GAMMA_CORRECTION) glEnable(GL_FRAMEBUFFER_SRGB);
    if(clearingMask)
        glClear(clearingMask);
}

}; //namespace display
}; //namespace rtac

