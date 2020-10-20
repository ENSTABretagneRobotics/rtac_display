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

#include <rtac_display/Display.h>
#include <rtac_display/PinholeView.h>
#include <rtac_display/PointCloudRenderer.h>
#include <rtac_display/NormalsRenderer.h>
using namespace rtac::display;

GLuint build_normals()
{
    GLuint normals;
    std::vector<float> data({-1,-1,-1,
                              1,-1,-1,
                              1, 1,-1,
                             -1, 1,-1,
                             -1,-1, 1,
                              1,-1, 1,
                              1, 1, 1,
                             -1, 1, 1});
    glGenBuffers(1, &normals);
    glBindBuffer(GL_ARRAY_BUFFER, normals);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*data.size(), data.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return normals;
}

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

    pcRenderer->set_normals(8, cubePoints.data());

    //auto nRenderer = NormalsRenderer::New(view3d);
    //nRenderer->set_normals(8, pcRenderer->points_, build_normals(), true);
    //display.add_renderer(nRenderer);

    float dangle = 0.001;
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


