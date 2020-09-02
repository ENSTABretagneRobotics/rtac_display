#ifndef _DEF_OPTIX_HELPERS_DISPLAY_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_H_

#include <iostream>
#include <memory>
#include <thread>

#include <GLFW/glfw3.h>

namespace optix_helpers {

class Display
{
    public:

    using Window = std::shared_ptr<GLFWwindow>;

    protected:
    
    static const std::string vertexShader;
    static const std::string fragmentShader;

    Window window_;

    void init_gl();

    public:

    Display(size_t width = 800, size_t height = 600,
            const std::string& title = "optix render");
    static void terminate();

    int should_close() const;
    void wait_for_close() const;
};

};

#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
