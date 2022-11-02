#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/types/Mesh.h>
using Mesh = rtac::types::Mesh<>;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/MeshRenderer.h>
#include <rtac_display/views/OrthoView.h>
#include <rtac_display/views/PinholeView.h>
using namespace rtac::display;
using Bounds = OrthoView::Bounds;
using Pose   = OrthoView::Pose;

int main()
{
    Display display;
    //auto view = OrthoView::Create(Bounds({-3,3,-3,3}));
    auto view = PinholeView::Create();
    auto axes = display.create_renderer<Renderer>(view);
    auto r = display.create_renderer<MeshRenderer>(view);

    view->look_at({0,0,0}, {5,4,3});

    while(!display.should_close()) {
        display.draw();
        this_thread::sleep_for(100ms);
    }
    return 0;
}
