#include <rtac_display/GLTexture.h>

namespace rtac { namespace display {

GLTexture::Ptr GLTexture::New()
{
    return Ptr(new GLTexture());
}

GLTexture::GLTexture() :
    shape_({0,0}),
    texId_(0),
    format_(GL_RGBA)
{
    this->init_texture();
    this->GLTexture::configure_texture();
}

GLTexture::~GLTexture()
{
    this->delete_texture();
}

void GLTexture::init_texture()
{
    if(!texId_)
        glGenTextures(1, &texId_);
}

void GLTexture::delete_texture()
{
    if(texId_)
        glDeleteTextures(1, &texId_);
    texId_ = 0;
    shape_ = Shape({0,0});
}

void GLTexture::configure_texture()
{
    glBindTexture(GL_TEXTURE_2D, texId_);

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
}

Shape GLTexture::shape() const
{
    return shape_;
}

GLuint GLTexture::gl_id() const
{
    return texId_;
}

GLint GLTexture::format() const
{
    return format_;
}

}; //namespace display
}; //namespace rtac


