#ifndef _DEF_RTAC_DISPLAY_COLOR_H_
#define _DEF_RTAC_DISPLAY_COLOR_H_

#include <iostream>
#include <iomanip>

#include <rtac_display/GLFormat.h>

namespace rtac { namespace display {

struct Color
{

template <typename T>
struct RGB
{
    T r; T g; T b;
};
using RGB8 = RGB<unsigned char>;
using RGBf = RGB<float>;

template <typename T>
struct RGBA
{
    T r; T g; T b; T a;
};
using RGBA8 = RGBA<unsigned char>;
using RGBAf = RGBA<float>;

}; //struct Color

template<>
struct GLFormat<Color::RGB<float>>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGB32F;
};

template<>
struct GLFormat<Color::RGBA<float>>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGBA32F;
};

template<>
struct GLFormat<Color::RGB<unsigned char>>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    //static constexpr GLenum InternalFormat = GL_RGB8UI;
    static constexpr GLenum InternalFormat = GL_RGB;
};

template<>
struct GLFormat<Color::RGBA<unsigned char>>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    //static constexpr GLenum InternalFormat = GL_RGBA8UI;
    static constexpr GLenum InternalFormat = GL_RGBA;
};

}; //namespace display
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::display::Color::RGB<T>& color)
{
    os << '(' << color.r << ',' << color.g << ',' << color.b << ')';
    return os;
}

template <>
inline std::ostream& operator<<(std::ostream& os, const rtac::display::Color::RGB<float>& color)
{
    auto p = os.precision();
    os << std::setprecision(3) << '(' 
       << color.r << ',' << color.g << ',' << color.b << ')'
       << std::setprecision(p);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::display::Color::RGBA<T>& color)
{
    os << '(' << color.r << ',' << color.g << ',' << color.b << ',' << color.a << ')';
    return os;
}

template <>
inline std::ostream& operator<<(std::ostream& os, const rtac::display::Color::RGBA<float>& color)
{
    auto p = os.precision();
    os << std::setprecision(3) << '(' 
       << color.r << ',' << color.g << ',' << color.b << ',' << color.a << ')'
       << std::setprecision(p);
    return os;
}

#endif //_DEF_RTAC_DISPLAY_COLOR_H_
