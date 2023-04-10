#include <rtac_base/containers/HostVector.h>
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/Scatter2D.h>
using namespace rtac::display;

int main()
{
    HostVector<float> x(360), y(360);
    for(unsigned int i = 0; i < x.size(); i++) {
        float theta = (2*M_PI*i) / x.size();
        x[i] = std::cos(theta);
        y[i] = std::sin(theta);
    }

    Display display;
    auto renderer = display.create_renderer<Scatter2D>(View::Create());
    renderer->set_data(x,y);

    while(!display.should_close()) {
        display.draw();
    }
}
