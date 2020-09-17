#ifndef _DEF_OPTIX_HELPERS_DISPLAY_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_H_

#include <iostream>
#include <memory>
#include <thread>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

//#include <glm/glm.hpp>
//#include <glm/gtx/transform.hpp>
//#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include <optix_helpers/Source.h>
#include <optix_helpers/display/utils.h>

namespace optix_helpers { namespace display {

class Display
{
    public:

    using Window = std::shared_ptr<GLFWwindow>;

    protected:
    
    static const Source vertexShader;
    static const Source fragmentShader;

    Window window_;
    GLuint displayProgram_;
    GLuint texId_;

    void init_texture();

    public:

    Display(size_t width = 800, size_t height = 600,
            const std::string& title = "optix render");
    void terminate();
    
    void set_image(size_t width, size_t height, const float* data);

    int should_close() const;
    void wait_for_close() const;

    void draw();
};

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
