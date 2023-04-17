#ifndef _DEF_RTAC_DISPLAY_COMMON_H_
#define _DEF_RTAC_DISPLAY_COMMON_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>
#include <GLFW/glfw3.h>

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
                << rtac::display::opengl_error_string(err)              \
                << "' (code:0x" << std::hex << err << std::dec << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while(0)                                                          \

namespace rtac { namespace display {

//inline bool check_gl(const std::string& location = "")
//{
//    GLenum errorCode;
//
//    errorCode = glGetError();
//    if(errorCode != GL_NO_ERROR)
//    {
//        std::ostringstream oss;
//        oss << "GL error : " << errorCode << ", \"" << gluErrorString(errorCode)
//            << "\". Tag : " << location;
//        //std::cout << oss.str() << std::endl;
//        throw std::runtime_error(oss.str());
//    }
//    return false;
//}

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

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_COMMON_H_
