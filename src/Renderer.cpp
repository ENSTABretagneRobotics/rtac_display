#include <optix_helpers/display/Renderer.h>

namespace optix_helpers { namespace display {

RendererObj::RendererObj(const Source& vertexShader, const Source& fragmentShader,
                         const View& view) :
    renderProgram_(create_render_program(vertexShader, fragmentShader)),
    view_(view)
{}

void RendererObj::draw()
{}

View RendererObj::view() const
{
    return view_;
}

}; //namespace display
}; //namespace optix_helpers

