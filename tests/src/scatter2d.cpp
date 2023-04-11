#include <rtac_base/containers/HostVector.h>
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/views/PlotView2D.h>
#include <rtac_display/renderers/Scatter2D.h>
using namespace rtac::display;

int main()
{
    HostVector<float> x(360), y(360);
    for(unsigned int i = 0; i < x.size(); i++) {
        float theta = (2*M_PI*i) / x.size();
        x[i] = 10.0*std::cos(theta);
        y[i] = 5.0*std::sin(theta);
    }

    Display display;
    auto data = PlotData2D::Create(x,y);
    auto view = PlotView2D::Create(data);
    view->enable_equal_aspect();
    auto renderer = display.create_renderer<Scatter2D>(view, data);

    while(!display.should_close()) {
        display.draw();
    }
}
