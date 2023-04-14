#include <rtac_base/types/SonarPing.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/FanRenderer.h>
using namespace rtac::display;

template <typename T>
__global__ void fill_ping(PingView2D<T> ping)
{
    ping(blockIdx.x, threadIdx.x) = ((blockIdx.x + threadIdx.x) & 0x1);
}

int main()
{
    Ping2D<float,CudaVector> p0(Linspace<float>(0.0f, 10.0f, 16),
                               CudaVector<float>::linspace(-0.25*3.14, 0.25*3.14, 16));
    fill_ping<<<p0.height(), p0.width()>>>(p0.view());
    cudaDeviceSynchronize();

    Display display;
    auto renderer = display.create_renderer<FanRenderer>(View::Create());
    renderer->set_ping(p0);

    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


