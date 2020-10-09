#include <iostream>
#include <vector>
#include <thread>
using namespace std;

#include <rtac_base/misc.h>
using FrameCounter = rtac::misc::FrameCounter;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/PointCloud.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;
using PointCloud = rtac::types::PointCloud<>;

#include <optix_helpers/display/Display.h>
#include <optix_helpers/display/PinholeView.h>
#include <optix_helpers/display/PointCloudRenderer.h>
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
    
    auto pcRenderer = PointCloudRenderer::New(view3d);
    display.add_renderer(pcRenderer);

    std::vector<float> cubePoints({-1,-1,-1,
                                    1,-1,-1,
                                    1, 1,-1,
                                   -1, 1,-1,
                                   -1,-1, 1,
                                    1,-1, 1,
                                    1, 1, 1,
                                   -1, 1, 1});
    //pcRenderer->set_points(8, cubePoints.data());
    PointCloud pc(8);
    pc[0] = PointCloud::PointType({-1,-1,-1});
    pc[1] = PointCloud::PointType({ 1,-1,-1});
    pc[2] = PointCloud::PointType({ 1, 1,-1});
    pc[3] = PointCloud::PointType({-1, 1,-1});
    pc[4] = PointCloud::PointType({-1,-1, 1});
    pc[5] = PointCloud::PointType({ 1,-1, 1});
    pc[6] = PointCloud::PointType({ 1, 1, 1});
    pc[7] = PointCloud::PointType({-1, 1, 1});
    cout << pc << endl;
    pcRenderer->set_points(pc);
    pcRenderer->set_pose(Pose({0,2,0}));


    float dangle = 0.01;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    FrameCounter counter;
    while(!display.should_close()) {
        view3d->set_pose(R * view3d->pose());
        
        display.draw();
        
        this_thread::sleep_for(10ms);
        cout << counter;
    }
    cout << endl;

    return 0;
}


