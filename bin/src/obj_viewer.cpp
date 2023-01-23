#include <iostream>
using namespace std;

#include <CLI/CLI.hpp>

#include <rtac_base/files.h>
#include <rtac_base/external/obj_codec.h>
using namespace rtac;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/MeshRenderer.h>
using namespace rtac::display;

int main(int argc, char** argv)
{
    CLI::App app{"A simple .obj file viewer"};
    
    std::string filename;
    app.add_option("filename", filename, "Path to .obj file to display.");
    
    app.parse(argc, argv);

    cout << "Opening file : " << filename << endl;

    external::ObjLoader parser(filename);
    parser.load_geometry();

    samples::Display3D display;

    for(auto mesh : parser.create_meshes<rtac::display::GLMesh>()) {
        auto renderer = display.create_renderer<MeshRenderer>(display.view());
        renderer->mesh() = mesh.second;
        renderer->set_texture(GLTexture::from_file(parser.material(mesh.first).map_Kd));
        renderer->set_render_mode(MeshRenderer::Mode::Textured);
    }

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


