#include <iostream>
using namespace std;

#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
#include <rtac_display/renderers/MeshRenderer.h>
using namespace rtac::display;
using Mesh = MeshRenderer::Mesh;

Mesh::Point operator-(const Mesh::Point& lhs, const Mesh::Point& rhs)
{
    return Mesh::Point({lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z});
}

int main(int argc, char** argv)
{
    CLI::App app("Simple renderer for 3D mesh in .ply format.");
    std::string filename("");
    auto farg = app.add_option("-f,--file", filename, ".ply file to display")
        ->required()
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);

    samples::Display3D display;
    display.add_renderer(Frame::New(Frame::Pose(), display.view()));

    auto mesh = Mesh::from_ply(filename);
    Mesh::Point min = mesh.points()[0];
    Mesh::Point max = mesh.points()[0];

    for(const auto& p : mesh.points()) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }
    cout << "min  : " << min << endl;
    cout << "max  : " << max << endl;
    cout << "diff : " << max - min << endl;

    auto meshRenderer = MeshRenderer::New(display.view());
    meshRenderer->set_mesh(mesh);
    display.add_renderer(meshRenderer);
    
    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
