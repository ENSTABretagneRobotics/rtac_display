#include <optix_helpers/Display.h>

namespace optix_helpers {

const std::string Display::vertexShader = R"(
#version 430 core

in vec2 points;
in vec2 texPoints;

uniform mat4 projection;
out vec2 texCoords;

int main()
{
    gl_Position = projection * vec4(points, 0.0, 1.0);
    texPointsOut = texPoints;
}

)";

const std::string Display::fragmentShader = R"(
#version 430 core

in vec2 texCoords;
uniform sampler2D tex;

out vec4 outColor;

int main()
{
    outColor = texture(tex, texCoords);
}

)";

Display::Display(size_t width, size_t height, const std::string& title) :
    window_(NULL)
{
    if(!glfwInit()) {
        throw std::runtime_error("GLFW initialization failure.");
    }
    window_ = Window(glfwCreateWindow(width, height, title.c_str(), NULL, NULL),
                     glfwDestroyWindow); //custom deleter
    if(!window_) {
        throw std::runtime_error("GLFW window creation failure.");
    }
    glfwMakeContextCurrent(window_.get());
}

void Display::terminate()
{
    glfwTerminate();
}

int Display::should_close() const
{
    glfwPollEvents();
    return glfwWindowShouldClose(window_.get()) > 0;
}

void Display::wait_for_close() const
{
    using namespace std;
    while(!this->should_close()) {
        std::this_thread::sleep_for(100ms);
    }
}

};

