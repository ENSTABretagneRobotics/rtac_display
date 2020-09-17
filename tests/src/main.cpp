#include <iostream>
#include <thread>
using namespace std;

#include <optix_helpers/display/Display.h>
using namespace optix_helpers::display;

int main()
{
    Display display;

    //display.wait_for_close();
    while(!display.should_close()) {
        display.draw();
        std::this_thread::sleep_for(100ms);
    }

    return 0;
}
