#include <iostream>
using namespace std;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/FrameInstances.h>
using namespace rtac::display;

int main()
{
    samples::Display3D display;
    display.create_renderer<Frame>(display.view());
    
    auto frames = display.create_renderer<FrameInstances>(display.view());

    Frame::Pose base({3.0f,0.0f,0.0f});
    int N = 10000;
    //int N = 10;
    for(int n = 0; n < N; n++) {
        frames->add_pose(Frame::Pose({0.0f,0.0f,0.0f},
                         Frame::Pose::Quat(Eigen::AngleAxisf((2.0*M_PI*n)/N, Eigen::Vector3f::UnitZ()))) * base);
    }
    
    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
