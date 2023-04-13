#include <iostream>
using namespace std;

#include <rtac_base/time.h>
using namespace rtac::time;

#include <rtac_display/Display.h>
#include <rtac_display/GLReductor.h>
using namespace rtac::display;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;
using namespace rtac;

#include "reductions.h"

int main()
{
    std::vector<float> data(1024*1024*64 + 1, 1.0f);
    //data[1001] = 1002;
    int N = 2000;
    //int N = 1010;
    Clock clock;
    double tGl, tGl2, tCuda;

    Display display;
    
    GLVector<float> glData(data);
    HostVector<float> glDump(glData.size());
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(glData);
        glDump = glData;
    }
    tGl = clock.now();
   
    GLReductor reductor;
    GLVector<float> glData2(data);
    HostVector<float> glDump2(glData2.size());
    auto p = reductor.sum_program("float");
    clock.reset();
    for(int n = 0; n < N; n++) {
        //reductor.reduce(glData2, p);
        GLReductor::reduce_in_place(glData2, p);
        glDump2 = glData2;
    }
    tGl2 = clock.now();
    
    CudaVector<float> cudaData(data);
    HostVector<float> cudaDump(cudaData.size());
    clock.reset();
    for(int n = 0; n < N; n++) {
        sum(cudaData);
        cudaDump = cudaData;
    }
    //cudaDump = cudaData;
    tCuda = clock.now();

    cout << "OpenGL time   : " << tGl << endl;
    cout << "OpenGL time 2 : " << tGl2 << endl;
    cout << "CUDA time     : " << tCuda << endl;

    cout << glData << endl;
    cout << glData2 << endl;
    cout << cudaData << endl;

    return 0;
}
