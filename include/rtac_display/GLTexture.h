#ifndef _DEF_RTAC_DISPLAY_GLTEXTURE_H_
#define _DEF_RTAC_DISPLAY_GLTEXTURE_H_

#include <GL/glew.h>
#include <GL/gl.h>

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

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
    
    Shape  shape()  const;
    GLuint gl_id()  const;
    GLint  format() const;

    template <typename T>
    void set_image(const Shape& shape, const T* data);
    template <typename T>
    void set_image(const Shape& shape, const GLVector<T>& data);
};

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
                                        
