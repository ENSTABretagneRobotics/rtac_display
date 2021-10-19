#ifndef _DEF_RTAC_DISPLAY_GLTEXTURE_H_
#define _DEF_RTAC_DISPLAY_GLTEXTURE_H_

#include <utility>

#include <GL/glew.h>
#include <GL/gl.h>

#include <rtac_base/types/Handle.h>
#include <rtac_base/files.h>

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
    void set_image(const Shape& shape, const T* data);
    template <typename T>
    void set_image(const Shape& shape, const GLVector<T>& data);

    void bind(GLenum target = GL_TEXTURE_2D);
    void unbind(GLenum target = GL_TEXTURE_2D);

    // various loaders.
    static Ptr from_ppm(const std::string& path);
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

    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, format_, shape.width, shape.height,
        0, format_, GLFormat<T>::Type, data);
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
    format_ = GLFormat<T>::PixelFormat;

    // ensuring no buffer bound to GL_PIXEL_UNPACK_BUFFER for data to be read
    // from CPU side memory.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, data.gl_id());

    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, format_, shape.width, shape.height,
        0, format_, GLFormat<T>::Type, 0);
    GL_CHECK_LAST();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    shape_ = shape;
}

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_GLTEXTURE_H_
                                        
