#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

#include <rtac_base/time.h>
using FrameCounter = rtac::time::FrameCounter;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Mesh.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;
using Mesh = rtac::types::Mesh<>;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
using namespace rtac::display;

int main()
{
    samples::Display3D display;
    display.create_renderer<Frame>(display.view());
    
    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


