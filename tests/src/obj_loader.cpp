#include <iostream>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/external/obj_codec.h>
using namespace rtac;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/MeshRenderer.h>
using namespace rtac::display;

int main()
{
    auto path = files::find_one(".*models3d/pyramide2_test01/.*obj");
    external::ObjLoader parser(path);

    parser.load_geometry();

    cout << "points   : " << parser.points().size()   << endl;
    cout << "uvs      : " << parser.uvs().size()      << endl;
    cout << "normals  : " << parser.normals().size()  << endl;
    cout << "vertices : " << parser.vertices().size() << endl;
    cout << "groups   : " << parser.faces().size()    << endl;

    samples::Display3D display;
    display.create_renderer<Frame>(display.view());
    
    for(auto mesh : parser.create_meshes<GLMesh>()) {
        auto renderer = display.create_renderer<MeshRenderer>(display.view());
        renderer->mesh() = mesh.second;
        
        renderer->set_texture(GLTexture::from_png(parser.material(mesh.first).map_Kd));
        renderer->set_render_mode(MeshRenderer::Mode::Textured);
    }

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
