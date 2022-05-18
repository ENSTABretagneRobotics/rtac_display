#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/MeshRenderer.h>
#include <rtac_display/text/TextRenderer.h>
using namespace rtac::display;

int main()
{
    int W = 1920, H = 1080;
    auto meshPath  = files::find_one(".*mummy_dtm_uav_withUV.ply");
    auto imagePath = files::find_one(".*mummy-orthoimage-fullResolution.ppm");
    cout << "Mesh path  : " << meshPath  << endl;
    cout << "Image path : " << imagePath << endl << flush;

    samples::Display3D display;
    display.controls()->look_at({0,0,0}, {5,4,3});
    display.create_renderer<Frame>(display.view());

    auto mesh = GLMesh::from_ply(meshPath, true);
    mesh->compute_normals();
    cout << "vertex count : " << mesh->points().size() << endl;
    cout << "face count   : " << mesh->faces().size() << endl;

    auto meshRenderer = display.create_renderer<MeshRenderer>(display.view());
    meshRenderer->mesh() = mesh;
    meshRenderer->set_texture(GLTexture::from_ppm(imagePath));

    std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    auto font = text::FontFace::Create(filename);
    font->set_char_size(12);

    auto label0 = display.create_renderer<text::TextRenderer>(
        display.view(), font, std::string("origin"));
    label0->set_anchor("top left");
    label0->set_text_color({1,1,1,1});
    auto label1 = display.create_renderer<text::TextRenderer>(
        display.view(), font, std::string("y"));
    label1->set_anchor("top left");
    label1->set_text_color({1,1,1,1});
    label1->origin()(1) = 1.0f;

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


