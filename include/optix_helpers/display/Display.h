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
    struct Shape
    {
        int width;
        int height;

        float ratio() const
        {
            return ((float)width/height);
        }

        void print(std::ostream& os)
        {
            os << "width : " << width
               << ", height : " << height
               << ", ratio : " << this->ratio()
               << "\n";
        }
    };

    protected:
    
    static const Source vertexShader;
    static const Source fragmentShader;

    Window window_;
    GLuint displayProgram_;
    GLuint texId_;
    Shape  imageSize_;

    void init_texture();

    public:

    Display(int width = 800, int height = 600,
            const std::string& title = "optix render");
    void terminate();
    
    void set_image(int width, int height, const float* data);

    int should_close() const;
    void wait_for_close() const;

    void draw();
};

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
