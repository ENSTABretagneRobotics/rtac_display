#include <iostream>
#include <thread>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/PointCloudRenderer.h>
using namespace rtac::display;
using Pose       = PinholeView::Pose;
using Quaternion = Pose::Quaternion;

#include <rtac_display/GLMappedPointer.h>
#include <rtac_display/GLVector.h>

GLVector<float> load_cube()
{
    GLVector<float> res(8*3);
    res.bind(GL_ARRAY_BUFFER);
    //auto data = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));

    GLMappedPointer<const float*> p(res.gl_id());
    //GLMappedPointer<float*> p(res.gl_id());
    auto data = p.get();

    //data[0] = -1; data[1] = -1; data[2] = -1; data += 3;
    //data[0] =  1; data[1] = -1; data[2] = -1; data += 3;
    //data[0] =  1; data[1] =  1; data[2] = -1; data += 3;
    //data[0] = -1; data[1] =  1; data[2] = -1; data += 3;

    //data[0] = -1; data[1] = -1; data[2] = 1; data += 3;
    //data[0] =  1; data[1] = -1; data[2] = 1; data += 3;
    //data[0] =  1; data[1] =  1; data[2] = 1; data += 3;
    //data[0] = -1; data[1] =  1; data[2] = 1;

    //glUnmapBuffer(GL_ARRAY_BUFFER);
    //res.unbind(GL_ARRAY_BUFFER);
    return res;
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

    auto cube = load_cube();
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
