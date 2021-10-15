#include <iostream>
#include <thread>
using namespace std;

#include <rtac_display/Display.h>
#include <rtac_display/text/FontFace.h>
using namespace rtac::display;

int main()
{
    std::string filename = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    Display display;

    auto font = text::FontFace::Create(filename);
    font->load_glyphs(0, 48);

    while(!display.should_close()) {
        display.draw();
        this_thread::sleep_for(20ms);
    }
    return 0;
}


