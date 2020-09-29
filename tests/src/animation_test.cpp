#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#include <rtac_base/types/Pose.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/samples/scenes.h>
using namespace optix_helpers::samples;

#include <optix_helpers/display/Display.h>
#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/ImageRenderer.h>
using namespace optix_helpers::display;

int main()
{
    int W = 1920, H = 1080;

    Display display;
    
    scenes::Scene0<RenderBufferGL> scene(W,H);

    scene.view()->look_at({0.0,0.0,0.0}, { 2.0, 5.0, 4.0});
    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    auto imageRenderer = ImageRenderer::New();
    display.add_renderer(imageRenderer);
    display.add_view(imageRenderer->view());

    int count = 0;
    auto t0 = chrono::high_resolution_clock::now();
    while(!display.should_close()) {
        scene.view()->set_pose(R * scene.view()->pose());
        
        scene.launch();
        imageRenderer->set_image(scene.render_buffer());
        display.draw();
        
        if(count >= 10) {
            auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> ellapsed = t1 - t0;
            cout << "Frame Rate : " << count / ellapsed.count() << "\r";
            count = 0;
            t0 = chrono::high_resolution_clock::now();
        }
        count++;
    }
    cout << endl;

    return 0;
}


