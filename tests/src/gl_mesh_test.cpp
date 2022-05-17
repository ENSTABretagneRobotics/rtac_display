#include <iostream>
using namespace std;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/GLMesh.h>
using namespace rtac::display;

int main()
{
    samples::Display3D display;
    display.create_renderer<Renderer>(display.view());

    auto mesh0 = GLMesh::cube();
    auto mesh1 = GLMesh::Create();

    cout << "mesh0 : " << *mesh0 << endl;
    cout << "mesh1 : " << *mesh1 << endl;

    *mesh1 = std::move(*mesh0);

    cout << "mesh0 : " << *mesh0 << endl;
    cout << "mesh1 : " << *mesh1 << endl;

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
