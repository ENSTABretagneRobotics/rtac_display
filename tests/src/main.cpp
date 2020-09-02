#include <iostream>
#include <thread>
using namespace std;

#include <optix_helpers/Display.h>
using namespace optix_helpers;

int main()
{
    Display display;

    display.wait_for_close();

    return 0;
}
