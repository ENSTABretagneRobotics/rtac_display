#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

#include <rtac_base/time.h>
#include <rtac_base/files.h>
#include <rtac_base/external/obj_codec.h>
using FrameCounter = rtac::time::FrameCounter;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Mesh.h>
using Pose = rtac::Pose<float>;
using Quaternion = rtac::Quaternion<float>;
using Mesh = rtac::Mesh<>;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/MeshRenderer.h>
#include <rtac_display/renderers/Frame.h>
using namespace rtac::display;

int main(int argc, char** argv)
{
    std::string filename = "";
    if(argc > 1) {
        filename = argv[1];
    }

    samples::Display3D display;
    
    auto origin = display.create_renderer<Frame>(display.view());

    if(filename.size() == 0) {
        auto meshRenderer = display.create_renderer<MeshRenderer>(display.view());
        auto mesh = GLMesh::icosahedron();
        mesh->compute_normals();
        GL_CHECK_LAST();
        meshRenderer->mesh() = mesh;
        meshRenderer->set_color({1,1,0,1});
    }
    else {
        rtac::external::ObjLoader parser(filename);
        parser.load_geometry();
        cout << parser << endl;
        cout << "bounding box :\n" << parser.bounding_box() << endl;
        for(const auto& m : parser.create_meshes<GLMesh>()) {
            auto renderer = display.create_renderer<MeshRenderer>(display.view());
            cout << *m.second << endl;
            renderer->mesh() = m.second;
            cout << "- GLMesh::bounding_box :\n" << m.second->bounding_box() << endl;
            renderer->set_color({1,1,0,1});
        }
    }

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


