#ifndef _DEF_RTAC_DISPLAY_TEXT_GLYPH_H_
#define _DEF_RTAC_DISPLAY_TEXT_GLYPH_H_

#include <iostream>

#include <rtac_base/types/Point.h>

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

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    types::Point2<long> bearing_;
    types::Point2<long> advance_;
    mutable GLTexture   texture_;
    GLuint              renderProgram_;

    Glyph(FT_Face face);

    public:

    // Disallowing Glyph copy
    Glyph(const Glyph&)            = delete;
    Glyph& operator=(const Glyph&) = delete;

    Glyph(Glyph&& other);
    Glyph& operator=(Glyph&& other);

    types::Point2<long> bearing() const;
    types::Point2<long> advance() const;
    const GLTexture&    texture() const;

    void draw() const;
};

}; //namespace text
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXT_GLYPH_H_
