#include <iostream>
#include <thread>
#include <vector>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/types/Point.h>
template <typename T>
using Point3 = rtac::types::Point3<T>;
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/renderers/ImageRenderer.h>
using namespace rtac::display;

std::vector<uint8_t> image_data(int width, int height)
{
    std::vector<uint8_t> res(3*width*height);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            unsigned int bw = (w + h) & 0x1;

            if(bw) {
                res[3*(width*h + w)]     = 0;
                res[3*(width*h + w) + 1] = 255;
                res[3*(width*h + w) + 2] = 255;
            }
            else {
                res[3*(width*h + w)]     = 255;
                res[3*(width*h + w) + 1] = 124;
                res[3*(width*h + w) + 2] = 0;
            }
        }
    }
    return res;
}

std::vector<Point3<uint8_t>> image_data_rgb(int width, int height)
{
    std::vector<Point3<uint8_t>> res(3*width*height);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            unsigned int bw = (w + h) & 0x1;
            if(bw) {
                res[width*h + w] = Point3<uint8_t>({0, 255, 255});
            }
            else {
                res[width*h + w] = Point3<uint8_t>({255, 124, 0});
            }
        }
    }
    return res;
}

std::vector<Point3<float>> image_data_rgbf(int width, int height)
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
    Display display;
    auto renderer = display.create_renderer<ImageRenderer>();
    
    unsigned int W = 4, H = 4;
    auto data0 = image_data(W,H);
    auto data1 = image_data_rgb(W,H);
    auto data2 = image_data_rgbf(W,H);
    GLVector<Point3<float>> data3(data2);

    // auto tex0 = GLTexture::New();
    // //tex0->set_image({W,H}, data0.data());
    // //tex0->set_image({W,H}, data1.data());
    // //tex0->set_image({W,H}, data2.data());
    // tex0->set_image({W,H}, data3);

    auto path = files::find_one(".*mummy-orthoimage-halfResolution.ppm");
    auto tex0 = GLTexture::from_ppm(path);

    //renderer->set_texture(tex0->shape(), tex0->gl_id());
    renderer->texture() = tex0;
    while(!display.should_close()) {
        display.draw();
        this_thread::sleep_for(10ms);
    }
    return 0;
}
