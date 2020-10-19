#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/types/Mesh.h>
using Mesh = rtac::types::Mesh<float, uint32_t, 3>;

#include <optix_helpers/display/Display.h>
#include <optix_helpers/display/MeshRenderer.h>
#include <optix_helpers/display/OrthoView.h>
#include <optix_helpers/display/PinholeView.h>
using namespace optix_helpers::display;
using Bounds = OrthoView::Bounds;
using Pose   = OrthoView::Pose;

int main()
{
    Display display;
    auto view = OrthoView::New(Bounds({-3,3,-3,3}));
    //auto view = PinholeView::New();
    auto axes = Renderer::New();
    axes->set_view(view);
    display.add_renderer(axes);

    auto r = MeshRenderer::New(view);
    r->set_mesh(Mesh::cube(1));
    display.add_renderer(r);

    view->look_at({0,0,0}, {5,4,3});

    while(!display.should_close()) {
        display.draw();
        this_thread::sleep_for(100ms);
    }
    return 0;
}
