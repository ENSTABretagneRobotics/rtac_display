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

//Renderer::Renderer()
//{}
//
//Renderer::Renderer(const Source& vertexShader, const Source& fragmentShader,
//                   const View& view) :
//    Handle<RendererObj>(vertexShader, fragmentShader, view)
//{}
//
//Renderer::Renderer(const std::shared_ptr<RendererObj>& obj) :
//    Handle<RendererObj>(obj)
//{}

}; //namespace display
}; //namespace optix_helpers

