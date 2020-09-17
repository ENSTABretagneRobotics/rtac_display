#include <iostream>
#include <thread>
using namespace std;

#include <optix_helpers/display/Display.h>
using namespace optix_helpers::display;

int main()
{
    Display display;

    const float data[12] = {0.0,0.0,1.0,
                            0.0,1.0,0.0,
                            0.0,1.0,0.0,
                            0.0,0.0,1.0};
    
    display.set_image(2,2,data);
    //display.wait_for_close();
    while(!display.should_close()) {
        display.draw();
        std::this_thread::sleep_for(100ms);
    }

    return 0;
}
