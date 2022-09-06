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
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/MeshRenderer.h>
#include <rtac_display/renderers/Frame.h>
using namespace rtac::display;

int main()
{
    int W = 1920, H = 1080;

    samples::Display3D display;
    
    auto origin = display.create_renderer<Frame>(display.view());

    auto meshRenderer = display.create_renderer<MeshRenderer>(display.view());
    auto mesh = GLMesh::icosahedron();
    mesh->compute_normals();
    GL_CHECK_LAST();
    meshRenderer->mesh() = mesh;
    meshRenderer->set_color({1,1,0,1});

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


