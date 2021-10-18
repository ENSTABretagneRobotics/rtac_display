#ifndef _DEF_RTAC_DISPLAY_UTILS_H_
#define _DEF_RTAC_DISPLAY_UTILS_H_

#include <iostream>
#include <sstream>
#include <memory>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <rtac_base/types/Shape.h>

#include <rtac_display/GLFormat.h>

#define GLFW_CHECK( call )                                              \
    do {                                                                \
        call;                                                           \
        unsigned int err = glfwGetError(NULL);                          \
        if(err != GLFW_NO_ERROR) {                                      \
            std::ostringstream oss;                                     \
            oss << "GLFW call '" << #call << "' failed '"               \
                << "' (code:" << err << ")\n"                           \
                << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while(0)                                                          \

#define GL_CHECK_LAST()                                                 \
    do {                                                                \
        GLenum err = glGetError();                                      \
        if(err != GL_NO_ERROR) {                                        \
            std::ostringstream oss;                                     \
            oss << "GL call failed '"                                   \
                << "' (code:0x" << std::hex << err << std::dec << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while(0)                                                          \

namespace rtac { namespace display {

using Shape = rtac::types::Shape<size_t>;

bool check_gl(const std::string& location = "");

GLuint compile_shader(GLenum shaderType, const std::string& source);

GLuint create_render_program(const std::string& vertexShaderSource,
                             const std::string& fragmentShaderSource);

GLuint create_compute_program(const std::string& computeShaderSource);

template <typename T>
inline void infer_gl_format(GLenum& format, GLenum& type)
{
    std::cerr << "Caution infer_gl_format<T> : "
              << "trying to infer data type from T. "
              << "using defaults GL_RED and GL_UNSIGNED_BYTE. "
              << "You should specialize ImageRenderer::infer_format "
              << "for your own types." << std::endl;
    format = GL_RED;
    type   = GL_UNSIGNED_BYTE;
}

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_UTILS_H_
