#include <iostream>
using namespace std;

#include <rtac_base/time.h>
using namespace rtac::time;

#include <rtac_display/Display.h>
using namespace rtac::display;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include "reductions.h"

int main()
{
    std::vector<float> data(1024*1024*32 + 1, 1.0f);
    int N = 100;
    Clock clock;
    double tGl, tCuda;

    Display display;
    
    GLVector<float> glData(data);
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(glData);
    }
    tGl = clock.now();
    
    
    
    DeviceVector<float> cudaData(data);
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(cudaData);
    }
    tCuda = clock.now();

    cout << "OpenGL time : " << tGl << endl;
    cout << "CUDA time   : " << tCuda << endl;

    cout << glData << endl;
    cout << cudaData << endl;

    return 0;
}
