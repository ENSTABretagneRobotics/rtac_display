#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#include <rtac_base/misc.h>
using FrameCounter = rtac::misc::FrameCounter;

#include <rtac_base/types/Pose.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/samples/scenes.h>
using namespace optix_helpers::samples;

#include <optix_helpers/display/Display.h>
#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/ImageRenderer.h>
#include <optix_helpers/display/View3D.h>
#include <optix_helpers/display/PinholeView.h>
using namespace optix_helpers::display;

int main()
{
    int W = 1920, H = 1080;

    Display display;
    
    auto renderer = Renderer::New();
    auto view3d = PinholeView::New();
    view3d->look_at({0,0,0}, {5,4,3});
    renderer->set_view(view3d);
    display.add_renderer(renderer);

    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));

    FrameCounter counter;
    //int count = 0;
    //auto t0 = chrono::high_resolution_clock::now();
    while(!display.should_close()) {
        view3d->set_pose(R * view3d->pose());
        
        display.draw();
        
        cout << counter;
        //if(count >= 10) {
        //    auto t1 = chrono::high_resolution_clock::now();
        //    chrono::duration<double> ellapsed = t1 - t0;
        //    cout << "Frame Rate : " << count / ellapsed.count() << "\r";
        //    count = 0;
        //    t0 = chrono::high_resolution_clock::now();
        //}
        //count++;
    }
    cout << endl;

    return 0;
}


