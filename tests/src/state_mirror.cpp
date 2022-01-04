#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/time.h>
using namespace rtac::time;

#include <rtac_display/GLState.h>
#include <rtac_display/GLStateMap.h>
#include <rtac_display/Display.h>
using namespace rtac::display;

int main()
{
    Display display;
    display.limit_frame_rate(10);
    display.disable_frame_counter();

    GLState    state0;
    GLStateMap state1;

    std::vector<GLenum> names;
    for(auto p : state1.stateMap_) {
        names.push_back(p.first);
    }

    Clock clock;
    auto t0 = clock.now();
    auto t1 = clock.now();
    bool res = false;
    int N = 50000000;
    while(!display.should_close()) {
        
        clock.reset();
        for(int n = 0; n < N; n++) {
            for(auto name : names) {
                res = res || state0.is_enabled(name);
            }
        }
        t0 = clock.now();
        
        clock.reset();
        for(int n = 0; n < N; n++) {
            for(auto name : names) {
                res = res || state1.is_enabled(name);
            }
        }
        t1 = clock.now();

        cout <<   "State    : " << t0
             << "\nStateMap : " << t1
             << "\nRatio    : " << t0 / t1 << endl << endl << flush;

        display.draw();
    }

    return 0;
}
