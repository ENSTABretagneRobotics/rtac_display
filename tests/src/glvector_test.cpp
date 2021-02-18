#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;

#include <rtac_display/Display.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/PointCloudRenderer.h>
using namespace rtac::display;
using Pose       = PinholeView::Pose;
using Quaternion = Pose::Quaternion;

#include <rtac_display/GLVector.h>

GLVector<float> load_cube_map()
{
    GLVector<float> v(8*3);

    {
        auto mappedPtr = v.map(true);
        float* data = mappedPtr;

        data[0] = -1; data[1] = -1; data[2] = -1; data += 3;
        data[0] =  1; data[1] = -1; data[2] = -1; data += 3;
        data[0] =  1; data[1] =  1; data[2] = -1; data += 3;
        data[0] = -1; data[1] =  1; data[2] = -1; data += 3;

        data[0] = -1; data[1] = -1; data[2] = 1; data += 3;
        data[0] =  1; data[1] = -1; data[2] = 1; data += 3;
        data[0] =  1; data[1] =  1; data[2] = 1; data += 3;
        data[0] = -1; data[1] =  1; data[2] = 1;

        //cout << v << endl; // should fail successfully because v already mapped. OK.
    }
    cout << v << endl; // should work OK

    return v;
}

GLVector<float> load_cube_bulk()
{
    std::vector<float> v(8*3);
    auto data = v.data();

    data[0] = -1; data[1] = -1; data[2] = -1; data += 3;
    data[0] =  1; data[1] = -1; data[2] = -1; data += 3;
    data[0] =  1; data[1] =  1; data[2] = -1; data += 3;
    data[0] = -1; data[1] =  1; data[2] = -1; data += 3;

    data[0] = -1; data[1] = -1; data[2] = 1; data += 3;
    data[0] =  1; data[1] = -1; data[2] = 1; data += 3;
    data[0] =  1; data[1] =  1; data[2] = 1; data += 3;
    data[0] = -1; data[1] =  1; data[2] = 1;

    DeviceVector<float> vd0(v);
    GLVector<float> vgl0(vd0);
    
    auto vd1 = vgl0.copy_to_cuda();

    return GLVector<float>(vd1);
}

int main()
{
    Display display;

    auto view = PinholeView::New();
    view->look_at({0,0,0}, {5,4,3});

    auto axes = Renderer::New();
    axes->set_view(view);
    display.add_renderer(axes);

    auto pcRenderer = PointCloudRenderer::New(view);
    display.add_renderer(pcRenderer);

    //auto cube = load_cube_bulk();
    auto cube = load_cube_map();
    cout << cube << endl;
    pcRenderer->set_points(cube.size(), cube.gl_id());

    float dangle = 0.01;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    while(!display.should_close()) {
        view->set_pose(R * view->pose());
        display.draw();
        this_thread::sleep_for(10ms);
    }
    return 0;
}
