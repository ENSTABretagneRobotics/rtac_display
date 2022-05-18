#include <iostream>
using namespace std;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/MeshRenderer.h>
#include <rtac_display/GLMesh.h>
using namespace rtac::display;

int main()
{
    samples::Display3D display;
    display.create_renderer<Frame>(display.view());

    auto mesh0 = GLMesh::cube();
    auto mesh1 = GLMesh::Create();

    cout << "mesh0 : " << *mesh0 << endl;
    cout << "mesh1 : " << *mesh1 << endl;

    *mesh1 = std::move(*mesh0);

    cout << "mesh0 : " << *mesh0 << endl;
    cout << "mesh1 : " << *mesh1 << endl;
    
    mesh1->compute_normals();
    cout << "mesh1 : " << *mesh1 << endl;

    auto renderer0 = display.create_renderer<MeshRenderer>(display.view());
    renderer0->mesh() = mesh1;
    renderer0->enable_normals_display();

    auto renderer1 = display.create_renderer<MeshRenderer>(display.view());
    renderer1->mesh() = mesh1;
    renderer1->set_pose(MeshRenderer::Pose({1.0,1.0,0.0}));
    renderer1->set_color({1.0,1.0,0.0,1.0});

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


