#ifndef _DEF_RTAC_DISPLAY_TEXT_GLYPH_H_
#define _DEF_RTAC_DISPLAY_TEXT_GLYPH_H_

#include <iostream>

#include <rtac_base/types/Point.h>

#include <rtac_display/Color.h>
#include <rtac_display/views/View.h>
#include <rtac_display/GLTexture.h>
#include <rtac_display/text/freetype.h>

namespace rtac { namespace display { namespace text {

// Forward declaration
class FontFace;

class Glyph
{
    public:
    
    // Only the FontFace type is allowed to create a new Glyph
    friend class FontFace;

    using Mat4 = View::Mat4;

    static const std::string vertexShader;
    static const std::string fragmentShaderFlat;
    static const std::string fragmentShaderSubPix;

    protected:
    
    types::Point2<float> bearing_;
    types::Point2<float> advance_;
    types::Point2<float> shape_;
    mutable GLTexture    texture_;

    GLuint renderProgramFlat_;
    GLuint renderProgramSubPix_;
    GLuint renderProgram_;

    Glyph(FT_GlyphSlot glyph);

    void load_bitmap(FT_GlyphSlot glyph);

    public:

    // Disallowing Glyph copy
    Glyph(const Glyph&)            = delete;
    Glyph& operator=(const Glyph&) = delete;

    Glyph(Glyph&& other);
    Glyph& operator=(Glyph&& other);

    types::Point2<float> bearing() const;
    types::Point2<float> advance() const;
    types::Point2<float> shape()   const;
    const GLTexture&     texture() const;

    void draw(const Mat4& mat = Mat4::Identity(),
              const Color::RGBAf& color = {0,0,0,0}) const;
};

}; //namespace text
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXT_GLYPH_H_
