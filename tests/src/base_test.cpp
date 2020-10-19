#include <iostream>
#include <thread>
using namespace std;

#include <rtac_display/Display.h>
using namespace rtac::display;

int main()
{
    Display display;

    const float data[12] = {0.0,0.0,1.0,
                            0.0,1.0,0.0,
                            0.0,1.0,0.0,
                            0.0,0.0,1.0};
    
    //display.wait_for_close();
    while(!display.should_close()) {
        display.draw();
        //std::this_thread::sleep_for(100ms);
    }

    return 0;
}
