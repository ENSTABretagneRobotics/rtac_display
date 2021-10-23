#ifndef _DEF_RTAC_DISPLAY_TEXT_RENDERER_H_
#define _DEF_RTAC_DISPLAY_TEXT_RENDERER_H_

#include <iostream>
#include <stdexcept>
#include <cmath>

#include <rtac_display/utils.h>
#include <rtac_display/views/View.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/text/FontFace.h>
#include <rtac_display/text/Glyph.h>

namespace rtac { namespace display { namespace text {

class TextRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<TextRenderer>;
    using ConstPtr = rtac::types::Handle<const TextRenderer>;
    using Mat4     = View::Mat4;
    using Vec4     = types::Vector4<float>;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    FontFace::ConstPtr font_;
    std::string        text_;
    GLTexture          texture_;
    Vec4               origin_; // full 3D space position.
    Color::RGBf        textColor_;
    Color::RGBAf       backColor_;
    
    TextRenderer(const FontFace::ConstPtr& font);

    public:

    static Ptr Create(const FontFace::ConstPtr& font,
                      const std::string& text);
    void set_text(const std::string& text, bool updateNow = true);
    void set_text_color(const Color::RGBf& color, bool updateNow = true);
    void set_back_color(const Color::RGBAf& color, bool updateNow = true);
    void update_texture();

    Shape compute_text_area(const std::string& text);

    FontFace::ConstPtr font() const;
    const std::string& text() const;
    const GLTexture&   texture() const;
    Mat4 view_matrix() const;
    Vec4& origin();
    const Vec4& origin() const;
    const Color::RGBf& text_color() const;
    const Color::RGBAf& back_color() const;

    virtual void draw();
};

}; //namespace text
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXT_RENDERER_H_
