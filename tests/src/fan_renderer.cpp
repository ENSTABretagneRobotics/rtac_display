#include <iostream>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/FanRenderer.h>
using namespace rtac::display;

using P4 = rtac::types::Point4<float>;

int main()
{
    Display display;

    auto renderer = display.create_renderer<FanRenderer>(View::New());
    renderer->set_geometry_degrees({-65,65}, {10,20});
    renderer->set_data(GLTexture::checkerboard({8,8}, 1.0f, 0.0f, 8));

    renderer->set_direction(FanRenderer::Direction::Up);
    //renderer->set_direction(FanRenderer::Direction::Down);
    //renderer->set_direction(FanRenderer::Direction::Left);
    //renderer->set_direction(FanRenderer::Direction::Right);

    while(!display.should_close()) {
        display.draw();
    }
    return 0;
}
