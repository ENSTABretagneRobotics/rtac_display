#ifndef _DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_
#define _DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_

#include <iostream>
#include <memory>

#include <rtac_display/GLTexture.h>

#include <rtac_display/text/freetype.h>
#include <rtac_display/text/Glyph.h>

namespace rtac { namespace display { namespace text {

class FontFace : public std::enable_shared_from_this<FontFace>
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
    
    void set_char_size(float pt, FT_UInt screenDpi = 102);
    void set_pixel_size(FT_UInt size);
    void load_glyphs();

    const GlyphMap& glyphs() const;
    const Glyph& glyph(uint8_t c) const;
    const FT_Face& face() const;
    
    // These return values in pixel (handle sub-pixels)
    float ascender() const;
    float descender() const;
    float baselineskip() const;
    float max_advance() const;
};

}; //namespace text
}; //namespace display
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os, const FT_Glyph_Metrics& metrics)
{
    const char* prefix = "\n- ";
    os << "FT_Size_Metrics :"
       << prefix << "width        : " << metrics.width        / 64.0f
       << prefix << "height       : " << metrics.height       / 64.0f
       << prefix << "horiBearingX : " << metrics.horiBearingX / 64.0f
       << prefix << "horiBearingY : " << metrics.horiBearingY / 64.0f
       << prefix << "horiAdvance  : " << metrics.horiAdvance  / 64.0f
       << prefix << "vertBearingX : " << metrics.vertBearingX / 64.0f
       << prefix << "vertBearingY : " << metrics.vertBearingY / 64.0f
       << prefix << "vertAdvance  : " << metrics.vertAdvance  / 64.0f;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const FT_Size_Metrics& metrics)
{
    const char* prefix = "\n- ";
    os << "FT_Size_Metrics :"
       << prefix << "x_ppem      : " << metrics.x_ppem
       << prefix << "y_ppem      : " << metrics.y_ppem
       << prefix << "x_scale     : " << metrics.x_scale / 65536.0
       << prefix << "y_scale     : " << metrics.y_scale / 65536.0
       << prefix << "ascender    : " << metrics.ascender / 64.0
       << prefix << "descender   : " << metrics.descender / 64.0
       << prefix << "height      : " << metrics.height / 64.0
       << prefix << "max_advance : " << metrics.max_advance / 64.0;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const FT_Bitmap_Size& bitmapSize)
{
    const char* prefix = "\n- ";
    os << "FT_Size_Metrics :"
       << prefix << "height : " << bitmapSize.height
       << prefix << "width  : " << bitmapSize.width
       << prefix << "size   : " << bitmapSize.size   / 64.0f
       << prefix << "x_ppem : " << bitmapSize.x_ppem / 64.0f
       << prefix << "y_ppem : " << bitmapSize.y_ppem / 64.0f;
    return os;
}

#endif //_DEF_RTAC_DISPLAY_TEXT_FONT_FACE_H_
