#include <iostream>
using namespace std;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/FrameInstances.h>
using namespace rtac::display;

int main()
{
    samples::Display3D display;
    display.add_renderer(Frame::New(Frame::Pose(),display.view()));

    auto frames = FrameInstances::New(display.view());
    display.add_renderer(frames);

    Frame::Pose base({3.0f,0.0f,0.0f});
    int N = 100;
    for(int n = 0; n < N; n++) {
        frames->add_pose(Frame::Pose({0.0f,0.0f,0.0f},
                         Frame::Pose::Quaternion(Eigen::AngleAxisf((2.0*M_PI*n)/N, Eigen::Vector3f::UnitZ()))) * base);
    }

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
