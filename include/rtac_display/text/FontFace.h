#ifndef _DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_
#define _DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_

#include <iostream>
#include <memory>

#include <rtac_display/GLTexture.h>

#include <rtac_display/text/freetype.h>
#include <rtac_display/text/Glyph.h>

namespace rtac { namespace display { namespace text {

class FontFace
{
    public:

    using Ptr      = std::shared_ptr<FontFace>;
    using ConstPtr = std::shared_ptr<const FontFace>;
    using GlyphMap = std::unordered_map<uint8_t, Glyph>;

    protected:

    Library::Ptr ft_;
    FT_Face      face_;
    GlyphMap     glyphs_;

    FontFace(const std::string& fontFilename,
             uint32_t faceIndex,
             const Library::Ptr& ftLibrary);

    public:

    static Ptr Create(const std::string& fontFilename,
                      uint32_t faceIndex = 0,
                      Library::Ptr ftLibrary = nullptr);

    void load_glyphs(FT_UInt pixelWidth, FT_UInt pixelHeight);
};

}; //namespace text
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_
