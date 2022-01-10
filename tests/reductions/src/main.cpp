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
    std::vector<float> data(1024*1024*64 + 1, 1.0f);
    int N = 1000;
    Clock clock;
    double tGl, tCuda;

    Display display;
    
    GLVector<float> glData(data);
    HostVector<float> glDump(glData.size());
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(glData);
    }
    glData.copy_to(glDump);
    tGl = clock.now();
    
    
    
    DeviceVector<float> cudaData(data);
    HostVector<float> cudaDump(cudaData.size());
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(cudaData);
    }
    cudaDump = cudaData;
    tCuda = clock.now();

    cout << "OpenGL time : " << tGl << endl;
    cout << "CUDA time   : " << tCuda << endl;

    cout << glData << endl;
    cout << cudaData << endl;

    return 0;
}
