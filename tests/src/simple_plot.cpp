#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
using namespace rtac;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/SimplePlot.h>
using namespace rtac::display;

HostVector<float> make_data(unsigned int N, float phase = 0)
{
    HostVector<float> data(N);
    for(unsigned int n = 0; n < N; n++) {
        data[n] = sin(4*2.0*M_PI*n / N + phase);
    }
    return data;
}

int main()
{
    Display display;
    auto renderer = display.create_renderer<SimplePlot>(View::Create());

    auto data = make_data(2048, 0.0);
    renderer->set_data(data);

    while(!display.should_close()) {
        display.draw();
    }
    return 0;
}
