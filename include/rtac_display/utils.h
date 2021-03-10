#ifndef _DEF_RTAC_DISPLAY_UTILS_H_
#define _DEF_RTAC_DISPLAY_UTILS_H_

#include <iostream>
#include <sstream>
#include <memory>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

namespace rtac { namespace display {

bool check_gl(const std::string& location = "");

GLuint compile_shader(GLenum shaderType, const std::string& source);

GLuint create_render_program(const std::string& vertexShaderSource,
                             const std::string& fragmentShaderSource);

GLuint create_compute_program(const std::string& computeShaderSource);

template <typename T>
void infer_gl_format(GLenum& format, GLenum& type)
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
