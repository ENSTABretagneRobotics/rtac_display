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

void write_char_ppm(char c, text::FontFace::Ptr font)
{
    std::ostringstream oss;
    oss << c << ".ppm";

    FT_Load_Char(font->face(), c, FT_LOAD_FORCE_AUTOHINT);
    auto glyph = font->face()->glyph;
    FT_Render_Glyph(glyph, FT_RENDER_MODE_LCD);

    // std::cout << "bitmap " << c << " : " << glyph->bitmap.width << "x"
    //                          << glyph->bitmap.rows << " p:"
    //                          << glyph->bitmap.pitch
    //                          << std::endl << std::endl;
    
    const char* prefix = "\n- ";
    std::cout << "bitmap " << c << " :" 
              << prefix << "bsize       : " << glyph->bitmap.width << "x"<< glyph->bitmap.rows
              << prefix << "bpitch      : " << glyph->bitmap.pitch
              << prefix << "horiBearing : " << glyph->metrics.horiBearingX / 64.0f << ", " << glyph->metrics.horiBearingY / 64.0f
              << prefix << "shape       : " << glyph->metrics.width  / 64.0f << ", " << glyph->metrics.height / 64.0f
              << prefix << "advance     : " << glyph->advance.x / 64.0f << ", " << glyph->advance.y
              << prefix << "linear adv  : " << glyph->linearHoriAdvance / 65536.0f << "x" << glyph->linearVertAdvance / 65536.0f
              << endl << endl;

    std::vector<Color::RGB8> data(glyph->bitmap.width*glyph->bitmap.rows / 3);
    auto itIn  = glyph->bitmap.buffer;
    for(int h = 0; h < glyph->bitmap.rows; h++) {
        for(int w = 0; w < glyph->bitmap.width; w += 3) {
            data[(glyph->bitmap.width*h + w) / 3].r = itIn[w];
            data[(glyph->bitmap.width*h + w) / 3].g = itIn[w + 1];
            data[(glyph->bitmap.width*h + w) / 3].b = itIn[w + 2];
        }
        itIn += glyph->bitmap.pitch;
    }
    write_ppm(oss.str(),
        font->face()->glyph->bitmap.width / 3,
        font->face()->glyph->bitmap.rows,
        (const char*)data.data());
}


int main()
{
    std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    //std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf";
    //std::string filename = "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf";
    Display display;
	display.disable_frame_counter();

    auto renderer = ImageRenderer::New();
    renderer->set_vertical_flip(false);
    //display.add_renderer(renderer);

    auto font = text::FontFace::Create(filename);
    font->set_char_size(10);
    font->set_char_size(12);
    //font->set_pixel_size(18);
    
    text::Glyph::Mat4 view = text::Glyph::Mat4::Identity();
    view(0,0) = 0.5f; view(1,1) = 0.5f;

    std::cout << font->face()->size->metrics << std::endl;
    std::cout << "units_per_EM : " << font->face()->units_per_EM << endl;
    std::cout << "ascender     : " << font->face()->ascender << endl;
    std::cout << "descender    : " << font->face()->descender << endl;
    std::cout << "height       : " << font->face()->height << endl << endl;

    //auto textRenderer = text::TextRenderer::Create(font, "Hello there !\nGeneral Kenobi !\n");
    std::ostringstream oss;
    oss << "Portez ce vieux whisky au juge blond qui fume." << 1234567890;
    for(int i = 0; i < 10; i++)
        oss << endl << "Portez ce vieux whisky au juge blond qui fume.";
    auto textRenderer = text::TextRenderer::Create(font, oss.str());
    display.add_renderer(textRenderer);
    textRenderer->origin()(0) = 0;
    textRenderer->origin()(1) = 0;
    textRenderer->set_anchor("top left");

    textRenderer->origin()(0) = -1;
    textRenderer->origin()(1) = -1;
    textRenderer->set_anchor("bottom left");
    
    // float c = 0.0;
    // display.set_clear_color({1,1,1,1});

    //float c = 0.7;
    float c = 1.0f;
    display.set_clear_color({56./255,12./255,42./255,1});
    //display.set_clear_color({0,0,0,1});
    //display.set_clear_color({48./255,10./255,36./255,1});

    textRenderer->set_text_color({c,c,c,1.0});

    //display.add_display_flags(DrawingSurface::GAMMA_CORRECTION);
    //display.remove_display_flags(DrawingSurface::GAMMA_CORRECTION);

    //write_char_pgm('n', font);
    //write_char_pgm('e', font);
    //write_char_pgm('h', font);
    //write_char_pgm('o', font);
    //write_char_pgm('P', font);

    write_char_ppm('n', font);
    write_char_ppm('e', font);
    write_char_ppm('h', font);
    write_char_ppm('o', font);
    write_char_ppm('P', font);
    write_char_ppm('i', font);
    write_char_ppm('l', font);
    write_char_ppm('t', font);


    cout << oss.str() << endl;
    auto glyph = font->glyphs().begin();
    while(!display.should_close()) {
        
        display.draw();
        //this_thread::sleep_for(250ms);
    }
    return 0;
}


