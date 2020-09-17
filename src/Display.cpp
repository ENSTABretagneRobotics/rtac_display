#include <optix_helpers/display/Display.h>

namespace optix_helpers { namespace display {

const Source Display::vertexShader = Source( R"(
#version 430 core

in vec2 points;
in vec3 color;
//in vec2 texPoints;

uniform mat4 projection;
//out vec2 texCoords;
out vec3 c;

void main()
{
    //gl_Position = projection * vec4(points, 0.0, 1.0);
    //texPointsOut = texPoints;
    gl_Position = vec4(points, 0.0, 1.0);
    c = color;
}
)", "vertex");

const Source Display::fragmentShader = Source(R"(
#version 430 core

//in vec2 texCoords;
//uniform sampler2D tex;
in vec3 c;

out vec4 outColor;

void main()
{
    //outColor = texture(tex, texCoords);
    outColor = vec4(c, 1.0);
}
)", "fragment");

Display::Display(size_t width, size_t height, const std::string& title) :
    window_(NULL),
    displayProgram_(0)
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
    
    // init glew (no gl function availabel if not done)
    GLenum initGlewStatus(glewInit());
    if(initGlewStatus != GLEW_OK)
        std::cout << "Failed to initialize glew" << std::endl;
    std::cout << glGetString(GL_VERSION) << std::endl;
    //if(GLEW_ARB_compute_shader)
    //    std::cout << "Compute shader ok !" << std::endl;

    //glClearColor(0.0,0.0,0.0,1.0);
    glClearColor(0.7,0.7,0.7,1.0);
    glViewport(0.0,0.0,width,height);
    displayProgram_ = create_render_program(vertexShader, fragmentShader);
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

void Display::draw()
{
    std::cout << "Drawing" << std::endl;
    float vertices[] = {-1.0,-1.0,
                         1.0,-1.0,
                         1.0, 1.0,
                        -1.0, 1.0};
                       
    float colors1[] = {1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       1.0, 1.0, 1.0,
                       0.0, 1.0, 0.0};
    unsigned int indexes[] = {0, 1, 2,
                              0, 2, 3};
    
    
    glClear(GL_COLOR_BUFFER_BIT);
    ////glLineWidth(11);
    ////glColor3f(0.7,0.7,0.0);

    glUseProgram(displayProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, colors1);
    glEnableVertexAttribArray(1);
    
    //glDrawElements(GL_LINE_STRIP, 3, GL_UNSIGNED_INT, indexes);
    //glDrawElements(GL_LINE_STRIP, 4, GL_UNSIGNED_INT, indexes);
    //glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, indexes);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    //glDrawArrays(GL_TRIANGLES, 0, 3);
    
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);

    glfwSwapBuffers(window_.get());
}

}; //namespace display
}; //namespace optix_helpers

