#include <iostream>
using namespace std;

#include <GL/glew.h>
#include <GL/gl.h>

#include <GLFW/glfw3.h>

void key_callback(GLFWwindow* window, int key, int scancode, int action, int modes)
{
    std::cout << "Got keyboard key : "
              << key << " " << scancode << " " << modes << " " << action
              << std::endl << std::flush;
}

bool should_close(GLFWwindow* window)
{
    glfwPollEvents();
    return glfwWindowShouldClose(window);
}

int main()
{
    unsigned int width = 800, height = 600;
    GLFWwindow* window_ = nullptr;
    if(!glfwInit()) {
        throw std::runtime_error("GLFW initialization failure.");
    }
    window_ = glfwCreateWindow(width, height, "input test", NULL, NULL);
    if(!window_) {
        throw std::runtime_error("GLFW window creation failure.");
    }
    glfwMakeContextCurrent(window_);
    
    // init glew (no gl function availabel if not done)
    GLenum initGlewStatus(glewInit());
    if(initGlewStatus != GLEW_OK)
        std::cout << "Failed to initialize glew" << std::endl;
    glClearColor(0.0,0.0,0.0,1.0);

    // // to measure fps
    // glfwSwapInterval(0);

    glViewport(0.0,0.0,width,height);

    glfwSetKeyCallback(window_, &key_callback);

    while(!should_close(window_));

    return 0;
}
