#ifndef _DEF_RTAC_DISPLAY_GLTEXTURE_H_
#define _DEF_RTAC_DISPLAY_GLTEXTURE_H_

#include <utility>

#include <GL/glew.h>
#include <GL/gl.h>

#include <rtac_base/types/Handle.h>
#include <rtac_base/files.h>
#ifdef RTAC_PNG
    #include <rtac_base/external/png_codec.h>
#endif //RTAC_PNG

#include <rtac_display/utils.h>
#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

/**
 * The main purpose of this class is to provide an easy interface to manipulate
 * OpenGL textures.
 *
 * Texture images can be loaded either from host memory or from a GLVector.
 * Memory management is hidden from the user. The texture internal format is
 * infered from the input data type at compile-time through the use of the
 * GLFormat structure.
 *
 * For now pixel data is stored as float32 only.
 */
class GLTexture
{
    public:

    using Ptr      = rtac::types::Handle<GLTexture>;
    using ConstPtr = rtac::types::Handle<const GLTexture>;

    enum WrapMode : GLint {
        Repeat        = GL_REPEAT,
        Mirror        = GL_MIRRORED_REPEAT,
        Clamp         = GL_CLAMP_TO_EDGE,
        ClampToEdge   = GL_CLAMP_TO_EDGE,
        ClampToBorder = GL_CLAMP_TO_BORDER,
    };
    enum FilterMode : GLint {
        Nearest = GL_NEAREST,
        Linear  = GL_LINEAR,
    };

    protected:
    
    Shape  shape_;
    GLuint texId_;
    GLint  format_;

    void init_texture();
    void delete_texture();
    virtual void configure_texture();

    public:
    
    static Ptr New();

    GLTexture();
    ~GLTexture();

    // disallowing copy but authorizing ressource move
    GLTexture(const GLTexture&)            = delete;
    GLTexture& operator=(const GLTexture&) = delete;

    GLTexture(GLTexture&& other);
    GLTexture& operator=(GLTexture&& other);
    
    Shape  shape()  const;
    GLuint gl_id()  const;
    GLint  format() const;
    size_t width() const;
    size_t height() const;

    template <typename T>
    void set_size(const Shape& shape);
    template <typename T>
    void set_image(const Shape& shape, GLint internalFormat,
                   GLenum pixelFormat, GLenum scalarType, const T* data);
    template <typename T>
    void set_image(const Shape& shape, const T* data);
    template <typename T>
    void set_image(const Shape& shape, const GLVector<T>& data);

    void bind(GLenum target = GL_TEXTURE_2D);
    void unbind(GLenum target = GL_TEXTURE_2D);
    
    // some configuration helpers
    void set_filter_mode(FilterMode mode);
    void set_filter_mode(FilterMode minMode, FilterMode magMode);
    void set_wrap_mode(WrapMode xyWrap);
    void set_wrap_mode(WrapMode xWrap, WrapMode yWrap);
    void set_wrap_mode(WrapMode xWrap, WrapMode yWrap, WrapMode zWrap);

    // various loaders.
    template <typename T>
    static std::vector<T> checkerboard_data(const Shape& shape, const T& c0, const T& c1,
                                            unsigned int oversampling = 1);
    template <typename T>
    static Ptr checkerboard(const Shape& shape, const T& c0, const T& c1,
                            unsigned int oversampling = 1);

    static Ptr from_ppm(const std::string& path);
    #ifdef RTAC_PNG
    static Ptr from_png(const std::string& path);
    #endif //RTAC_PNG
};

/**
 * Set texture size without data initialization
 *
 * The texture format is infered using the template type **T** and a template
 * specialization of rtac::display::GLFormat. See rtac::display::GLformat
 * documentation for more information.
 *
 * @param shape Dimensions of the texture {width,height}. Texture width must be even.
 * @param data  Pixel data to upload to the texture.
 */
template <typename T>
void GLTexture::set_size(const Shape& shape)
{
    format_ = GLFormat<T>::PixelFormat;

    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, format_, shape.width, shape.height,
        0, format_, GLFormat<T>::Type, 0);
    GL_CHECK_LAST();
    glBindTexture(GL_TEXTURE_2D, 0);

    shape_ = shape;
}

/**
 * Set texture image data from host memory.
 *
 * @param shape Dimensions of the texture {width,height}. Texture width must be even.
 * @param data  Pixel data to upload to the texture.
 */
template <typename T>
void GLTexture::set_image(const Shape& shape,
                          GLint internalFormat,
                          GLenum pixelFormat,
                          GLenum scalarType,
                          const T* data)
{
    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, shape.width, shape.height,
        0, pixelFormat, scalarType, data);
    GL_CHECK_LAST();
    glBindTexture(GL_TEXTURE_2D, 0);

    shape_ = shape;
}

/**
 * Set texture image data from host memory.
 *
 * The texture format is infered using the template type **T** and a template
 * specialization of rtac::display::GLFormat. See rtac::display::GLformat
 * documentation for more information.
 *
 * @param shape Dimensions of the texture {width,height}. Texture width must be even.
 * @param data  Pixel data to upload to the texture.
 */
template <typename T>
void GLTexture::set_image(const Shape& shape, const T* data)
{
    format_ = GLFormat<T>::PixelFormat;
    using Format = GLFormat<T>;

    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, Format::InternalFormat, shape.width, shape.height,
        0, Format::PixelFormat, Format::Type, data);
    GL_CHECK_LAST();
    glBindTexture(GL_TEXTURE_2D, 0);

    shape_ = shape;
}

/**
 * Set texture image data from an OpenGL Buffer Object.
 *
 * The texture format is infered using the template type **T** and a template
 * specialization of rtac::display::GLFormat. See rtac::display::GLformat
 * documentation for more information.
 *
 * @param shape Dimensions of the texture {width,height}. Texture width must be even.
 * @param data  GLVector containing the pixel data.
 */
template <typename T>
void GLTexture::set_image(const Shape& shape, const GLVector<T>& data)
{
    if(shape.area() > data.size()) {
        throw std::runtime_error("Too few data for requested texture size");
    }
    using Format = GLFormat<T>;
    format_ = GLFormat<T>::PixelFormat;

    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, data.gl_id());

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, Format::InternalFormat, shape.width, shape.height,
        0, Format::PixelFormat, Format::Type, 0);
    GL_CHECK_LAST();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    shape_ = shape;
}

inline void GLTexture::set_filter_mode(FilterMode mode)
{
    this->bind(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mode);
}

inline void GLTexture::set_filter_mode(FilterMode minMode, FilterMode magMode)
{
    this->bind(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode);
}

inline void GLTexture::set_wrap_mode(WrapMode xyWrap)
{
    this->bind(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, xyWrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, xyWrap);
}

inline void GLTexture::set_wrap_mode(WrapMode xWrap, WrapMode yWrap)
{
    this->bind(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, xWrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, yWrap);
}

inline void GLTexture::set_wrap_mode(WrapMode xWrap, WrapMode yWrap, WrapMode zWrap)
{
    this->bind(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, xWrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, yWrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, zWrap);
}

template <typename T>
std::vector<T> GLTexture::checkerboard_data(const Shape& shape, const T& c0, const T& c1,
                                            unsigned int oversampling)
{
    std::vector<T> data(shape.area()*oversampling*oversampling);
    for(unsigned int i = 0; i < shape.height*oversampling; i++) {
        for(unsigned int j = 0; j < shape.width*oversampling; j++) {
            if((i / oversampling + j / oversampling) & 0x1)
                data[shape.width*oversampling*i + j] = c0;
            else
                data[shape.width*oversampling*i + j] = c1;
        }
    }
    
    return data;
}

template <typename T>
GLTexture::Ptr GLTexture::checkerboard(const Shape& shape, const T& c0, const T& c1,
                                       unsigned int oversampling)
{
    auto data = checkerboard_data(shape, c0, c1, oversampling);
    auto tex = GLTexture::New();
    tex->set_image({shape.width*oversampling, shape.height*oversampling}, data.data());
    return tex;
}

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_GLTEXTURE_H_
                                        
