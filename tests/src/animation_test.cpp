#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#include <rtac_base/types/Pose.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/samples/scenes.h>
#include <optix_helpers/samples/utils.h>
using namespace optix_helpers::samples;

#include <optix_helpers/display/Display.h>
using namespace optix_helpers::display;

int main()
{
    int W = 1920, H = 1080;

    Display display;
    
    GLuint renderVbo = display.create_buffer(3*sizeof(float)*W*H);

    //scenes::Scene0 scene(W,H);
    scenes::Scene0 scene(W,H, renderVbo);

    std::vector<float> data(3*W*H);
    for(int i = 0; i < 3*W*H; i++) data[i] = 0.7;

    scene.view()->look_at({0.0,0.0,0.0}, { 2.0, 5.0, 4.0});
    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));

    int count = 0;
    auto t0 = chrono::high_resolution_clock::now();
    while(!display.should_close()) {
        scene.view()->set_pose(R * scene.view()->pose());
        
        scene.launch();
        //scene.view()->write_data(reinterpret_cast<uint8_t*>(data.data()));
        //display.set_image(W,H,data.data());
        display.set_buffer(W,H,renderVbo);
        display.draw();
        
        if(count >= 10) {
            auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> ellapsed = t1 - t0;
            cout << "Frame Rate : " << count / ellapsed.count() << "\r";
        }
        count++;
    }

    return 0;
}


