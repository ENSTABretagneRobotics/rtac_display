#include <optix_helpers/display/Renderer.h>

namespace optix_helpers { namespace display {

RendererObj::RendererObj(const Source& vertexShader, const Source& fragmentShader,
                         const View& view) :
    renderProgram_(create_render_program(vertexShader, fragmentShader)),
    view_(view)
{}

void RendererObj::set_screen_size(const Shape& screen)
{
    view_->update_projection(screen);
}

void RendererObj::draw()
{}

}; //namespace display
}; //namespace optix_helpers

