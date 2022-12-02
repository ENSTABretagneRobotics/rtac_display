#include <iostream>
#include <vector>
#include <thread>
using namespace std;

#include <rtac_base/time.h>
using FrameCounter = rtac::time::FrameCounter;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/PointCloud.h>
using Pose = rtac::Pose<float>;
using Quaternion = rtac::Quaternion<float>;
using PointCloud = rtac::PointCloud<>;

#include <rtac_display/Display.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/MeshRenderer.h>
using namespace rtac::display;

struct PointTest {
    float x;
    float y;
    float z;
    float w;
};

int main()
{
    int W = 1920, H = 1080;

    Display display;
    
    auto view3d   = PinholeView::Create();
    auto renderer = display.create_renderer<Renderer>(view3d);
    view3d->look_at({0,0,0}, {5,4,3});
    
    auto pcRenderer = display.create_renderer<MeshRenderer>(view3d);

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
    auto mesh = GLMesh::Create();
    *mesh = pc;
    pcRenderer->mesh() = mesh;
    pcRenderer->set_render_mode(MeshRenderer::Mode::Points);
    pcRenderer->set_pose(Pose::from_translation(Pose::Vec3({0,2,0})));


    float dangle = 0.01;
    auto R = Pose::from_quaternion(Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
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


