#include <iostream>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/FanRenderer.h>
using namespace rtac::display;

using P4 = rtac::types::Point4<float>;

GLTexture::Ptr create_texture(int W, int H)
{
    std::vector<float> data(W*H);
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            data[W*h + w] = ((float)w) / (W - 1);
        }
    }
    auto texture = GLTexture::Create();
    texture->set_image({W,H}, data.data());
    return texture;
}


int main()
{
    Display display;

    auto renderer = display.create_renderer<FanRenderer>(View::Create());
    renderer->set_geometry_degrees({-65,65}, {0,20});

    //auto texture = create_texture(16,16);
    //auto texture = GLTexture::checkerboard({16,16}, 1.0f, 0.0f);
    auto data = GLTexture::checkerboard_data({16,16}, 1.0f, 0.0f);
    for(auto& v : data) v *= 2;
    for(auto& v : data) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    auto texture = GLTexture::Create();
    texture->set_image({16,16}, data.data());
    texture->set_filter_mode(GLTexture::FilterMode::Linear);
    //texture->set_filter_mode(GLTexture::FilterMode::Nearest);
    texture->set_wrap_mode(GLTexture::WrapMode::Clamp);

    renderer->set_data(texture);

    std::vector<float> bearings(32);
    for(int i = 0; i < bearings.size(); i++) {
        bearings[i] = 65.0f*M_PI/180.0f
                    //* sin(M_PI*(((float)i) / (bearings.size() - 1) - 0.5f));
                    * tan(0.5f*M_PI*(((float)i) / (bearings.size() - 1) - 0.5f));
    }
    renderer->set_bearings(bearings.size(), bearings.data());

    renderer->set_direction(FanRenderer::Direction::Up);
    //renderer->set_direction(FanRenderer::Direction::Down);
    //renderer->set_direction(FanRenderer::Direction::Left);
    //renderer->set_direction(FanRenderer::Direction::Right);

    //renderer->set_value_range({0.0f,2.0f});
    //renderer->set_value_range({0.2f,1.0f});
    //renderer->set_value_range({1.0f, 0.0f});

    while(!display.should_close()) {
        display.draw();
    }
    return 0;
}
