#include <rtac_base/containers/HostVector.h>
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/views/PlotView2D.h>
#include <rtac_display/renderers/SimplePlot2D.h>
using namespace rtac::display;

int main()
{
    HostVector<float> x(360), y(360);
    for(unsigned int i = 0; i < x.size(); i++) {
        float theta = (2*M_PI*i) / x.size();
        x[i] = 100.0*std::cos(theta);
        y[i] = 50.0*std::sin(theta);
    }

    HostVector<float> z(200);
    for(unsigned int i = 0; i < z.size(); i++) {
        float theta = 10*(2*M_PI*i) / z.size();
        z[i] = 20.0*std::cos(theta);
    }

    Display display;
    auto view = PlotView2D::Create();
    view->enable_equal_aspect();
    view->scaling().enable_memory();

    auto data = PlotData2D::Create(x,y);
    auto renderer = display.create_renderer<SimplePlot2D>(view, data);
    renderer->set_draw_mode(GL_POINTS);

    auto dataZ = PlotData2D::Create(z);
    auto rendererZ = display.create_renderer<SimplePlot2D>(view, dataZ);
    rendererZ->set_draw_mode(GL_LINE_STRIP);

    std::cout << "data  : " << data->x_range()  << ", " << data->y_range()  << std::endl;
    std::cout << "dataZ : " << dataZ->x_range() << ", " << dataZ->y_range() << std::endl;

    view->update_ranges(data);
    view->update_ranges(dataZ);

    while(!display.should_close()) {
        display.draw();
    }
}
