#ifndef _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
#define _DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_

#include <rtac_base/types/Handle.h>

#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/ImageView.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

class ImageRenderer : public Renderer
{
    protected:

    void init_texture();

    public:

    using Ptr      = rtac::types::Handle<ImageRenderer>;
    using ConstPtr = rtac::types::Handle<const ImageRenderer>;

    using Mat4  = ImageView::Mat4;
    using Shape = ImageView::Shape;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:

    GLuint texId_;
    ImageView::Ptr imageView_;

    public:

    static Ptr New();

    ImageRenderer();
    ~ImageRenderer();
    
    virtual void draw();
    
    // Setting image from client memory
    void set_image(const Shape& imageSize, const void* data,
                   GLenum format, GLenum type);
    //void set_image(const Shape& imageSize, const void* data); // legacy
    void set_gray_image(const Shape& imageSize, const void* data);
    void set_rgb_image(const Shape& imageSize, const void* data);

    void set_image(const Shape& imageSize, GLuint buffer,
                   GLenum format = GL_RGB, GLenum type = GL_FLOAT);
    
    template <typename T>
    static void infer_format(GLenum& format, GLenum& type);
    template <typename T>
    void set_image(const Shape& imageSize, const T* data);
    template <typename T>
    void set_image(const Shape& imageSize, const GLVector<T>& data);
};

// Implementation
template <typename T>
void ImageRenderer::infer_format(GLenum& format, GLenum& type)
{
    std::cerr << "Caution ImageRenderer::infer_format<T> : "
              << "trying to infer data type from T. "
              << "using defaults GL_RED and GL_UNSIGNED_BYTE. "
              << "You should specialize ImageRenderer::infer_format "
              << "for your own types." << std::endl;
    format = GL_RED;
    type   = GL_UNSIGNED_BYTE;
}

template <typename T>
void ImageRenderer::set_image(const Shape& imageSize, const T* data)
{
    GLenum format, type;
    ImageRenderer::infer_format<T>(format, type);
    this->set_image(imageSize, data, format, type);
}

template <typename T>
void ImageRenderer::set_image(const Shape& imageSize, const GLVector<T>& data)
{
    GLenum format, type;
    if(data.size() < imageSize.area()) {
        throw std::runtime_error("GLVector not big enough for image");
    }
    ImageRenderer::infer_format<T>(format, type);
    this->set_image(imageSize, data.gl_id(), format, type);
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_IMAGE_RENDERER_H_
