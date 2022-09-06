#ifndef _DEF_RTAC_DISPLAY_UTILS_H_
#define _DEF_RTAC_DISPLAY_UTILS_H_

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Rectangle.h>

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
            oss << "GL call failed '" << opengl_error_string(err)       \
                << "' (code:0x" << std::hex << err << std::dec << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while(0)                                                          \

namespace rtac { namespace display {

using Shape = rtac::types::Shape<size_t>;
using Rect  = rtac::types::Rectangle<size_t>;

inline bool check_gl(const std::string& location = "")
{
    GLenum errorCode;

    errorCode = glGetError();
    if(errorCode != GL_NO_ERROR)
    {
        std::ostringstream oss;
        oss << "GL error : " << errorCode << ", \"" << gluErrorString(errorCode)
            << "\". Tag : " << location;
        //std::cout << oss.str() << std::endl;
        throw std::runtime_error(oss.str());
    }
    return false;
}

inline constexpr const char* opengl_error_string(GLenum code)
{
    switch(code) {
        default:                   return "GL_UNKNOWN_ERROR";     break;
        case GL_NO_ERROR:          return "GL_NO_ERROR";          break;
        case GL_INVALID_ENUM:      return "GL_INVALID_ENUM";      break;
        case GL_INVALID_VALUE:     return "GL_INVALID_VALUE";     break;
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION"; break;
        case GL_OUT_OF_MEMORY:     return "GL_OUT_OF_MEMORY";     break;
        case GL_STACK_UNDERFLOW:   return "GL_STACK_UNDERFLOW";   break;
        case GL_STACK_OVERFLOW:    return "GL_STACK_OVERFLOW";    break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: 
            return "GL_INVALID_FRAMEBUFFER_OPERATION";            break;
    }
}

GLuint compile_shader(GLenum shaderType, const std::string& source);

GLuint create_render_program(const std::string& vertexShaderSource,
                             const std::string& fragmentShaderSource);

GLuint create_compute_program(const std::string& computeShaderSource);
GLuint create_compute_program(const std::vector<std::string>& computeShaderSources);

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
