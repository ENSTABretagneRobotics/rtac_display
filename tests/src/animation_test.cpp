#include <iostream>
#include <vector>
using namespace std;

#include <optix_helpers/samples/scenes.h>
#include <optix_helpers/samples/utils.h>
using namespace optix_helpers::samples;

#include <optix_helpers/display/Display.h>
using namespace optix_helpers::display;

int main()
{
    int W = 1920, H = 1080;
    scenes::Scene0 scene(W,H);
    scene.launch();
    //utils::display(scene.view()->render_buffer());

    std::vector<float> data(3*W*H);
    scene.view()->write_data(reinterpret_cast<uint8_t*>(data.data()));

    Display display;

    display.set_image(W,H,data.data());
    while(!display.should_close()) {
        display.draw();
        //std::this_thread::sleep_for(100ms);
    }

    return 0;
}


