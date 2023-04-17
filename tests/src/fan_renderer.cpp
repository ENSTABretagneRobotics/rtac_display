#include <rtac_base/types/SonarPing.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;

#include <rtac_display/Display.h>
#include <rtac_display/renderers/FanRenderer.h>
using namespace rtac::display;

template <typename T> inline
void fill_ping(PingView2D<T> ping, unsigned int oversampling = 1)
{
    for(unsigned int h = 0; h < ping.height(); h++) {
        for(unsigned int w = 0; w < ping.width(); w++) {
            ping(h,w) = ((h/oversampling + w/oversampling) & 0x1);
        }
    }
}

inline HostVector<float> make_bearings(unsigned int N)
{
    HostVector<float> bearings(N);
    float thetaMax = 3.0f;
    float a = (M_PI / 180.0f) / tan(thetaMax * 0.5f);
    for(int i = 0; i < bearings.size(); i++) {
        float x = ((float)i) / (bearings.size() - 1) - 0.5f;
        bearings[i] = 65.0f * a * tan(thetaMax*x);
    }
    return bearings;
}

int main()
{
    //unsigned int N = 4;
    //unsigned int oversampling = 1;
    //unsigned int N = 16;
    //unsigned int oversampling = 4;
    //unsigned int N = 32;
    //unsigned int oversampling = 4;
    unsigned int N = 256;
    unsigned int oversampling = 4;
    Ping2D<float> p0(Linspace<float>(0.0f, 10.0f, N),
                     //HostVector<float>::linspace(-0.25*3.14, 0.25*3.14, N));
                     make_bearings(N));
    fill_ping(p0.view(), oversampling);

    Display display;
    auto renderer = display.create_renderer<FanRenderer>(View::Create());
    renderer->set_ping(p0);
    //renderer->disable_bearing_map();

    auto program = renderer->render_program();
    GLint uniformCount = -1, maxSize = -1;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS,           &uniformCount);
    glGetProgramiv(program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxSize);

    for(int i = 0; i < uniformCount; i++) {
        std::vector<char> name(maxSize);
        GLsizei strLength = -1;
        GLint   uniformSize = -1;
        GLenum  uniformType;
        glGetActiveUniform(program, i, maxSize, 
                           &strLength, &uniformSize, &uniformType, (GLchar*) name.data());
        std::cout << strLength << std::endl;
        name[strLength] = '\0';
        std::cout << "Uniform '" << name.data() << "' :"
                  << "\n- size : " << uniformSize
                  << "\n- type : " << uniformType << std::endl;
    }


    while(!display.should_close()) {
        display.draw();
    }

    return 0;
}


