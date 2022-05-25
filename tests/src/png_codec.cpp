#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/ImageRenderer.h>
using namespace rtac::display;

int main()
{
    auto path = files::find_one(".*\\.png");
    cout << "Using image : " << path << endl;

    Display display;
    auto renderer = display.create_renderer<ImageRenderer>();

    renderer->texture() = GLTexture::from_png(path);

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}
