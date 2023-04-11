#include <iostream>
using namespace std;

#include <rtac_display/Scaling.h>
using namespace rtac;
using namespace rtac::display;

int main()
{
    Scaling1D scaling;
    std::cout << scaling << std::endl;

    scaling.update(Bounds<float>(3,8));
    std::cout << scaling << std::endl;

    scaling.set_origin(-10);
    std::cout << scaling << std::endl;

    scaling.update({20.0f, 50.0f});
    std::cout << scaling << std::endl;

    scaling.disable_origin();
    scaling.update({20.0f, 50.0f});
    std::cout << scaling << std::endl;

    scaling.enable_memory();
    scaling.update({21.0f, 35.0f});
    std::cout << scaling << std::endl;

    scaling.update({0.0f, 60.0f});
    std::cout << scaling << std::endl;

    scaling.set_limits({10.0f, 20.0f});
    std::cout << scaling << std::endl;

    scaling.enable_origin();
    std::cout << scaling << std::endl;

    return 0;
}
