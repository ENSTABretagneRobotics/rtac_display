#ifndef _DEF_RTAC_DISPLAY_GLSL_TYPE_H_
#define _DEF_RTAC_DISPLAY_GLSL_TYPE_H_

#include <type_traits>

#include <rtac_display/GLFormat.h>

namespace rtac { namespace display {

template <unsigned int Size>
struct GLSLTypeFloat
{
    static_assert(Size > 0 && Size <= 4, "Invalid GLSL type size");

    struct Float1 { static constexpr const char* value = "float"; };
    struct Float2 { static constexpr const char* value = "vec2"; };
    struct Float3 { static constexpr const char* value = "vec3"; };
    struct Float4 { static constexpr const char* value = "vec4"; };
    
    static constexpr const char* value = std::conditional<Size == 1, Float1,
                                typename std::conditional<Size == 2, Float2,
                                typename std::conditional<Size == 3, Float3, Float4
                                >::type>::type>::type::value;
};

template <unsigned int Size>
struct GLSLTypeInt
{
    static_assert(Size > 0 && Size <= 4, "Invalid GLSL type size");

    struct Int1 { static constexpr const char* value = "int"; };
    struct Int2 { static constexpr const char* value = "ivec2"; };
    struct Int3 { static constexpr const char* value = "ivec3"; };
    struct Int4 { static constexpr const char* value = "ivec4"; };
    
    static constexpr const char* value = std::conditional<Size == 1, Int1,
                                typename std::conditional<Size == 2, Int2,
                                typename std::conditional<Size == 3, Int3, Int4
                                >::type>::type>::type::value;
};

template <unsigned int Size>
struct GLSLTypeUint
{
    static_assert(Size > 0 && Size <= 4, "Invalid GLSL type size");

    struct Uint1 { static constexpr const char* value = "uint"; };
    struct Uint2 { static constexpr const char* value = "uvec2"; };
    struct Uint3 { static constexpr const char* value = "uvec3"; };
    struct Uint4 { static constexpr const char* value = "uvec4"; };
    
    static constexpr const char* value = std::conditional<Size == 1, Uint1,
                                typename std::conditional<Size == 2, Uint2,
                                typename std::conditional<Size == 3, Uint3, Uint4
                                >::type>::type>::type::value;
};

/**
 * The purpose of the GLSLType struct is to infer a GLSL native type string
 * from a GLFormat type struct.
 *
 * This allows to write generic glsl shader code (analogous to C++ templates)
 * which will be automatically adapted to a user type. (See GLReductor for an
 * example).
 */
template <typename T>
struct GLSLType
{
    using Format                       = GLFormat<T>;
    using Scalar                       = typename Format::Scalar;
    static constexpr unsigned int Size = Format::Size;

    static_assert(Size > 0 && Size <= 4, "Invalid GLSL type size");
    static_assert(std::is_integral<Scalar>::value
               || std::is_same<Scalar,float>::value
               || std::is_same<Scalar,double>::value,
               "Invalid GLSL scalar type");

    static constexpr const char* value = std::conditional<
        std::is_integral<Scalar>::value,
            typename std::conditional<std::is_signed<Scalar>::value,
                GLSLTypeInt<Size>, GLSLTypeUint<Size>>::type,
            GLSLTypeFloat<Size>
        >::type::value;
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GLSL_TYPE_H_
