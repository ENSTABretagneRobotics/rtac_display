#ifndef _DEF_RTAC_DISPLAY_GL_FORMAT_H_
#define _DEF_RTAC_DISPLAY_GL_FORMAT_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <rtac_base/types/Point.h>

namespace rtac { namespace display {

// These are used to automatically infer GLenum values from custom types
template <typename T>
struct GLFormatError : std::false_type {};

template <typename T>
struct GLFormat {
    // This will trigger a compile error if GLFormat template is instanciated
    // and the specific specialization was not already defined
    static_assert(GLFormatError<T>::value, "Could not infer GL type");
};

// Some specialization used in this library. Other specialization may be
// defined by the user.
template<>
struct GLFormat<float>
{
    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_FLOAT;
};

template<>
struct GLFormat<uint8_t>
{
    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;
};

template<>
struct GLFormat<types::Point3<float>>
{
    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_FLOAT;
};

template<>
struct GLFormat<types::Point3<uint32_t>>
{
    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_UNSIGNED_INT;
};

template<>
struct GLFormat<types::Point3<uint8_t>>
{
    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;
};

template<>
struct GLFormat<types::Point2<float>>
{
    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_FLOAT;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_FORMAT_H_
