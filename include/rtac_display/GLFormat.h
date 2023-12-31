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

/**
 * The purpose of the GLFormat struct is to infer at compile time some
 * information on a pixel (or a point) type.
 *
 * GLFormat<PixelT> is not usable as is. A template specialization **MUST** be
 * declared for each type it will be used with. Trying to use GLFormat with a
 * type **T** without an explicit GLFormat<T> specialization will result in a
 * compilation error. See GLFormat.h for examples of template specializations.
 *
 * For example, when pixel data is to be uploaded to a texture, the user
 * provide a non-typed pixel data array to the OpenGL API. Since the pixel
 * array is not-typed (void*), the user must provide to the OpenGL API the
 * pixel format (RED, RGB, RGBA...) and the data type (float, byte..) of the
 * input data and the internal pixel format of the texture (see more
 * [here](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml)).
 *
 * Instead of filling this information by hand, the GLFormat allows all of the
 * pixel information to be "infered" from a custom pixel type. This allows for
 * more generic functions.
 * 
 * Example with a user type PixelRGB_float :
 * - A specialization of GLFormat is declared
 * \verbatim
 struct PixelRGB_float
 {
     float r;
     float g;
     float b;
 };

 template<>
 struct rtac::display::GLFormat<PixelRGB_float>
 {
     static constexpr unsigned int Size  = 3;        // PixelRGB_float has 3 color components.
     static constexpr GLenum PixelFormat = GL_RGB;   // OpenGL RGB format enum value.
     static constexpr GLenum Type        = GL_FLOAT; // Underlying scalar type.
 };
 \endverbatim
 * Caution : the specialization must be declared inside the namespace rtac::display.
 *
 * - Then the GLFormat object can be used in a generic templated function :
 * \verbatim
 template <typename T>
 void fill_texture(int width, int height, const T* data)
 {
      glTexImage2D(GL_TEXTURE_2D, 0,
                   GLFormat<T>::PixelFormat, // internalFormat parameter inferred from T.
                   width, height, 0,
                   GLFormat<T>::PixelFormat, // format parameter inferred from T.
                   GLFormat<T>::Type,        // scalar type inferred from T.
                   (const void*)data);
 }
 
 void main()
 {
     ...
     PixelRGB_float* data = new ...;
     ...
     fill_texture(1024,1024,data);
 }
 \endverbatim
 *
 * If the specialization GLFormat<PixelRGB_float> was not declared, a compile
 * error will be triggered.
 */
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
    using Scalar = float;

    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_R32F;
};

template<>
struct GLFormat<rtac::Point2<float>>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RG32F;
};

template<>
struct GLFormat<rtac::Point3<float>>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGB32F;
};

template<>
struct GLFormat<rtac::Point4<float>>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGBA32F;
};

template<>
struct GLFormat<int32_t>
{
    using Scalar = int32_t;

    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_INT;

    static constexpr GLenum InternalFormat = GL_R32I;
};

template<>
struct GLFormat<rtac::Point2<int32_t>>
{
    using Scalar = int32_t;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_INT;

    static constexpr GLenum InternalFormat = GL_RG32I;
};

template<>
struct GLFormat<rtac::Point3<int32_t>>
{
    using Scalar = int32_t;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_INT;

    static constexpr GLenum InternalFormat = GL_RGB32I;
};

template<>
struct GLFormat<rtac::Point4<int32_t>>
{
    using Scalar = int32_t;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_INT;

    static constexpr GLenum InternalFormat = GL_RGBA32I;
};

template<>
struct GLFormat<uint32_t>
{
    using Scalar = uint32_t;

    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_UNSIGNED_INT;

    static constexpr GLenum InternalFormat = GL_R32UI;
};

template<>
struct GLFormat<rtac::Point2<uint32_t>>
{
    using Scalar = uint32_t;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_UNSIGNED_INT;

    static constexpr GLenum InternalFormat = GL_RG32UI;
};

template<>
struct GLFormat<rtac::Point3<uint32_t>>
{
    using Scalar = uint32_t;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_UNSIGNED_INT;

    static constexpr GLenum InternalFormat = GL_RGB32UI;
};

template<>
struct GLFormat<rtac::Point4<uint32_t>>
{
    using Scalar = uint32_t;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_UNSIGNED_INT;

    static constexpr GLenum InternalFormat = GL_RGBA32UI;
};

template<>
struct GLFormat<char>
{
    using Scalar = char;

    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_BYTE;

    static constexpr GLenum InternalFormat = GL_R8I;
};

template<>
struct GLFormat<rtac::Point2<char>>
{
    using Scalar = char;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_BYTE;

    static constexpr GLenum InternalFormat = GL_RG8I;
};

template<>
struct GLFormat<rtac::Point3<char>>
{
    using Scalar = char;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_BYTE;

    static constexpr GLenum InternalFormat = GL_RGB8I;
};

template<>
struct GLFormat<rtac::Point4<char>>
{
    using Scalar = char;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_BYTE;

    static constexpr GLenum InternalFormat = GL_RGBA8I;
};

template<>
struct GLFormat<unsigned char>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 1;
    static constexpr GLenum PixelFormat = GL_RED;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    static constexpr GLenum InternalFormat = GL_RED;
    //static constexpr GLenum InternalFormat = GL_R8UI;
};

template<>
struct GLFormat<rtac::Point2<unsigned char>>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    //static constexpr GLenum InternalFormat = GL_RG8UI;
    static constexpr GLenum InternalFormat = GL_RG;
};

template<>
struct GLFormat<rtac::Point3<unsigned char>>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    static constexpr GLenum InternalFormat = GL_RGB;
};

template<>
struct GLFormat<rtac::Point4<unsigned char>>
{
    using Scalar = unsigned char;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_UNSIGNED_BYTE;

    //static constexpr GLenum InternalFormat = GL_RGBA8UI;
    static constexpr GLenum InternalFormat = GL_RGBA;
};


// template <>
// struct GLFormat<double>
// {
//     using Scalar = double;
// 
//     static constexpr unsigned int Size  = 1;
//     static constexpr GLenum PixelFormat = GL_RED;
//     static constexpr GLenum Type        = GL_DOUBLE;
// };
// 
// template <>
// struct GLFormat<rtac::Point2<double>>
// {
//     using Scalar = double;
// 
//     static constexpr unsigned int Size  = 2;
//     static constexpr GLenum PixelFormat = GL_RG;
//     static constexpr GLenum Type        = GL_DOUBLE;
// };
// 
// template <>
// struct GLFormat<rtac::Point3<double>>
// {
//     using Scalar = double;
// 
//     static constexpr unsigned int Size  = 3;
//     static constexpr GLenum PixelFormat = GL_RGB;
//     static constexpr GLenum Type        = GL_DOUBLE;
// };
// 
// template <>
// struct GLFormat<rtac::Point4<double>>
// {
//     using Scalar = double;
// 
//     static constexpr unsigned int Size  = 4;
//     static constexpr GLenum PixelFormat = GL_RGBA;
//     static constexpr GLenum Type        = GL_DOUBLE;
// };

}; //namespace display
}; //namespace rtac

#ifdef RTAC_CUDA_ENABLED

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace display {

template<>
struct GLFormat<float2>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 2;
    static constexpr GLenum PixelFormat = GL_RG;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RG32F;
};

template<>
struct GLFormat<float3>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 3;
    static constexpr GLenum PixelFormat = GL_RGB;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGB32F;
};

template<>
struct GLFormat<float4>
{
    using Scalar = float;

    static constexpr unsigned int Size  = 4;
    static constexpr GLenum PixelFormat = GL_RGBA;
    static constexpr GLenum Type        = GL_FLOAT;

    static constexpr GLenum InternalFormat = GL_RGBA32F;
};
#endif //RTAC_DISPLAY_CUDA

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_FORMAT_H_
