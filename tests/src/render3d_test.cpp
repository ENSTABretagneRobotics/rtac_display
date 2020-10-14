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
    
    scenes::Scene0<RenderBufferGL> scene(W,H);

    scene.view()->look_at({0.0,0.0,0.0}, { 2.0, 5.0, 4.0});
    float dangle = 0.001;
    //float dangle = 0.003;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    auto imageRenderer = ImageRenderer::New();
    display.add_renderer(imageRenderer);

    auto renderer = Renderer::New();
    auto view3d = PinholeView::New();
    view3d->look_at({0,0,0}, {5,4,3});
    renderer->set_view(view3d);
    //display.add_renderer(renderer);
    
    FrameCounter counter;
    while(!display.should_close()) {
        scene.view()->set_pose(R * scene.view()->pose());
        view3d->set_pose(scene.view()->pose());
        
        scene.launch();
        imageRenderer->set_image(scene.render_buffer());
        display.draw();
        
        cout << counter;
    }
    cout << endl;

    return 0;
}


