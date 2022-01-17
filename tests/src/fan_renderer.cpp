#include <iostream>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/FanRenderer.h>
using namespace rtac::display;

int main()
{
    Display display;
    auto renderer = display.create_renderer<FanRenderer>(View::New());

    //renderer->set_geometry_degrees({-40,40}, {0,20});
    //renderer->set_geometry_degrees({-65,65}, {0,20});
    renderer->set_geometry_degrees({-170,170}, {0,20});
    
    while(!display.should_close()) {
        display.draw();
    }
    return 0;
}
