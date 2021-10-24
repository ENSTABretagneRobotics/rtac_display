#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>
#include <rtac_display/text/FontFace.h>
#include <rtac_display/text/TextRenderer.h>
using namespace rtac::display;

void write_char_pgm(char c, text::FontFace::Ptr font)
{
    std::ostringstream oss;
    oss << c << ".pgm";

    FT_Load_Char(font->face(), c, FT_LOAD_RENDER);
    write_pgm(oss.str(),
        font->face()->glyph->bitmap.width,
        font->face()->glyph->bitmap.rows,
        (const char*)font->face()->glyph->bitmap.buffer);
}

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
    //font->set_pixel_size(18);
    font->set_char_size(12);
    
    text::Glyph::Mat4 view = text::Glyph::Mat4::Identity();
    view(0,0) = 0.5f; view(1,1) = 0.5f;

    std::cout << font->face()->size->metrics << std::endl;
    std::cout << "units_per_EM : " << font->face()->units_per_EM << endl;
    std::cout << "ascender     : " << font->face()->ascender << endl;
    std::cout << "descender    : " << font->face()->descender << endl;
    std::cout << "height       : " << font->face()->height << endl << endl;

    //auto textRenderer = text::TextRenderer::Create(font, "Hello there !\nGeneral Kenobi !\n");
    std::ostringstream oss;
    for(int i = 0; i < 20; i++)
        oss << "Portez ce vieux whisky au juge blond qui fume." << endl;
    auto textRenderer = text::TextRenderer::Create(font, oss.str());
    display.add_renderer(textRenderer);
    textRenderer->origin()(0) = -1;
    textRenderer->origin()(1) = -1;
    
    // display.set_clear_color({1,1,1,1});
    // textRenderer->set_text_color({0,0,0});
    // //display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
    // display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

    //display.set_clear_color({0,0,0,0});
    //display.set_clear_color({48./255,10./255,36./255,1});
    display.set_clear_color({56./255,12./255,42./255,1});
    //textRenderer->set_text_color({1,1,1});
    float c = 0.7;
    textRenderer->set_text_color({c,c,c});
    //textRenderer->set_text_color({0.5,0.5,0.5});
    //display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
    //display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

   // display.set_clear_color({0,0,0,0});
   // textRenderer->set_text_color({0.1,0.3,0.1});
   // //display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
   // display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

    write_char_pgm('n', font);
    write_char_pgm('e', font);
    write_char_pgm('h', font);
    write_char_pgm('o', font);
    write_char_pgm('P', font);

    cout << oss.str() << endl;
    auto glyph = font->glyphs().begin();
    while(!display.should_close()) {
        
        display.draw();
        //this_thread::sleep_for(250ms);
    }
    return 0;
}


