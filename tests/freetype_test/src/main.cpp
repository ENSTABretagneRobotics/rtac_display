#include <iostream>
#include <thread>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>
#include <rtac_display/text/FontFace.h>
using namespace rtac::display;

int main()
{
    std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    Display display;
    auto renderer = ImageRenderer::New();
    display.add_renderer(renderer);

    auto font = text::FontFace::Create(filename);
    font->load_glyphs(0, 48);
    
    auto glyph = font->glyphs().begin();
    while(!display.should_close()) {
        
        display.draw();
        glClearColor(1.0,1.0,1.0,1.0);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        
        glyph->second.draw({1,0,0});
        glyph++;
        if(glyph == font->glyphs().end())
            glyph = font->glyphs().begin();

        glfwSwapBuffers(display.window().get());

        this_thread::sleep_for(250ms);
    }
    return 0;
}


