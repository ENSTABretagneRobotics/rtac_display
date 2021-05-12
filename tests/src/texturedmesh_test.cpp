#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

#include <rtac_base/time.h>
#include <rtac_base/files.h>
#include <rtac_base/types/Point.h>
using FrameCounter = rtac::time::FrameCounter;
template <typename T>
using Point2 = rtac::types::Point2<T>;
template <typename T>
using Point3 = rtac::types::Point3<T>;
using namespace rtac;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Mesh.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;
using Mesh = rtac::types::Mesh<>;

#include <rtac_display/samples/Display3D.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/renderers/TexturedMeshRenderer.h>
using namespace rtac::display;

GLVector<Point3<float>> image_data_rgbf(int width, int height)
{
    std::vector<Point3<float>> res(3*width*height);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            unsigned int bw = (w + h) & 0x1;
            if(bw) {
                res[width*h + w] = Point3<float>({0, 1, 1});
            }
            else {
                res[width*h + w] = Point3<float>({1, 0.5, 0});
            }
        }
    }
    return res;
}

int main()
{
    int W = 1920, H = 1080;

    samples::Display3D display;
    display.controls()->look_at({0,0,0}, {5,4,3});
    
    auto renderer = Renderer::New();
    // auto view3d = PinholeView::New();
    // view3d->look_at({0,0,0}, {5,4,3});
    // renderer->set_view(view3d);
    display.add_renderer(renderer);

    auto meshRenderer = TexturedMeshRenderer<>::New(display.view());

    auto mesh = Mesh::cube(0.5);
    Mesh::PointVector points(0);
    for(auto f : mesh.faces()) {
        points.push_back(mesh.point(f.x));
        points.push_back(mesh.point(f.y));
        points.push_back(mesh.point(f.z));
    }
    std::vector<Point2<float>> uvs({
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({1,0}),
        Point2<float>({0,0}), Point2<float>({0,1}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,0}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({0,1}),
        Point2<float>({0,0}), Point2<float>({1,0}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({0,1}),
        Point2<float>({0,0}), Point2<float>({1,0}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({0,1}),
        Point2<float>({0,0}), Point2<float>({1,0}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({0,1}),
        Point2<float>({0,0}), Point2<float>({1,0}), Point2<float>({1,1}),
        Point2<float>({0,0}), Point2<float>({1,1}), Point2<float>({0,1})
    });
    *meshRenderer->points() = points;
    *meshRenderer->uvs()    = uvs;

    // meshRenderer->texture()->set_image({4,4}, image_data_rgbf(4,4));
    auto path = files::find_one(".*mummy-orthoimage-halfResolution.ppm");
    meshRenderer->texture() = GLTexture::from_ppm(path);

    // *meshRenderer->points() = mesh.points();
    // *meshRenderer->faces()  = mesh.faces();

    meshRenderer->set_pose(Pose({0,0,3}));
    display.add_renderer(meshRenderer);

    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    FrameCounter counter;
    while(!display.should_close()) {
        //view3d->set_pose(R * view3d->pose());
        
        display.draw();
        
        //cout << counter;
        this_thread::sleep_for(10ms);
    }
    cout << endl;

    return 0;
}


