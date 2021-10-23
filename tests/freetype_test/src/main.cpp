#include <iostream>
#include <thread>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>
#include <rtac_display/text/FontFace.h>
#include <rtac_display/text/TextRenderer.h>
using namespace rtac::display;

int main()
{
    std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    //std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf";
    Display display;
	display.disable_frame_counter();

    auto renderer = ImageRenderer::New();
    renderer->set_vertical_flip(false);
    //display.add_renderer(renderer);

    auto font = text::FontFace::Create(filename);
    font->load_glyphs(0, 18);
    
    text::Glyph::Mat4 view = text::Glyph::Mat4::Identity();
    view(0,0) = 0.5f; view(1,1) = 0.5f;

    std::cout << font->face()->size->metrics << std::endl;
    std::cout << "units_per_EM : " << font->face()->units_per_EM << endl;
    std::cout << "ascender     : " << font->face()->ascender << endl;
    std::cout << "descender    : " << font->face()->descender << endl;
    std::cout << "height       : " << font->face()->height << endl << endl;

    auto textRenderer = text::TextRenderer::Create(font, "Hello there !\nGeneral Kenobi !");
    display.add_renderer(textRenderer);
    textRenderer->origin()(0) = -1;
    textRenderer->origin()(1) = 0;
    
    display.set_clear_color({1,1,1,1});
    textRenderer->set_text_color({0,0,0});
    display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
    // display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

    // display.set_clear_color({0,0,0,0});
    // textRenderer->set_text_color({1,1,1});
    // display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
    // //display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

    auto glyph = font->glyphs().begin();
    while(!display.should_close()) {
        
        display.draw();
        //this_thread::sleep_for(250ms);
    }
    return 0;
}


