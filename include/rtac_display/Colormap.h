#ifndef _DEF_RTAC_DISPLAY_COLORMAP_H_
#define _DEF_RTAC_DISPLAY_COLORMAP_H_

#include <vector>

#include <rtac_display/Color.h>
#include <rtac_display/GLTexture.h>

namespace rtac { namespace display {

class Colormap
{
    public:

    using Ptr      = rtac::Handle<Colormap>;
    using ConstPtr = rtac::Handle<const Colormap>;

    protected:

    GLTexture texture_;

    template <typename T>
    Colormap(const std::vector<T>& rgbaData);

    public:
    
    template <typename T>
    static Ptr Create(const std::vector<T>& rgbaData);

    GLTexture& texture();
    const GLTexture& texture() const;
};

template <typename T>
Colormap::Colormap(const std::vector<T>& rgbaData)
{
    texture_.set_image({rgbaData.size() / 4, 1},
                       reinterpret_cast<const Color::RGBA<T>*>(rgbaData.data()));
    texture_.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    texture_.unbind();
}

template <typename T>
Colormap::Ptr Colormap::Create(const std::vector<T>& rgbaData)
{
    return Ptr(new Colormap(rgbaData));
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_COLORMAP_H_
