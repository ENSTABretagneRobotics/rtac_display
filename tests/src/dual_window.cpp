#include <iostream>
using namespace std;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
using namespace rtac::display;

// OBSERVATION :
// Sharing of display data between windows is working, but API design disallow
// a single renderer and different views. API should be refactored and
// renderers splitted into a drawer and a handle, with the view in the handle.

int main()
{
    samples::Display3D display0;
    display0.enable_frame_counter();
    display0.view()->look_at({0,0,0}, {5,4,3});
    
    auto renderer = display0.create_renderer<Frame>(display0.view(), Frame::Pose());

    //samples::Display3D display1; // this is not working (as expected)
    samples::Display3D display1(display0.context());
    display1.view()->look_at({0,0,0}, {5,4,3});
    display1.add_render_item(renderer, display1.view());

    while(!display0.should_close() && !display1.should_close()) {
        display0.draw();
        display1.draw();
    }

    return 0;
}
