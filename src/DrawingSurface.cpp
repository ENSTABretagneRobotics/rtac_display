#include <rtac_display/DrawingSurface.h>

namespace rtac { namespace display {

DrawingSurface::DrawingSurface(const Shape& shape) :
    Renderer(Renderer::vertexShader, Renderer::fragmentShader, View::New())
{
    this->view_->set_screen_size(shape);
}

/**
 * Instanciate a new DrawingSurface.
 *
 * @param shape initial {width,height} of the DrawingSurface.
 *
 * @return a shared pointer to the newly instanciated DrawingSurface.
 */
DrawingSurface::Ptr DrawingSurface::New(const Shape& shape)
{
    return Ptr(new DrawingSurface(shape));
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

/**
 * Append a rtac::display::Renderer to the list of handled renderers.
 *
 * The view associated to the renderer will automatically be added to the
 * handled views via DrawingSurface::add_view method.
 */
void DrawingSurface::add_renderer(const Renderer::Ptr& renderer)
{
    renderers_.push_back(renderer);
    views_.push_back(renderer->view());
}

/**
 * Update all handled views with the current display size and draw all the
 * handled renderers after clearing the display area.
 */
void DrawingSurface::draw()
{
    Shape shape = this->view()->screen_size();
    for(auto view : views_) {
        view->set_screen_size(shape);
    }

    glViewport(0,0,shape.width,shape.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for(auto renderer : renderers_) {
        if(renderer) {
            renderer->draw();
        }
    }
}

}; //namespace display
}; //namespace rtac

