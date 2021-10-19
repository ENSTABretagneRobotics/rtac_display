#include <rtac_display/text/FontFace.h>

namespace rtac { namespace display { namespace text {

FontFace::FontFace(const std::string& fontFilename,
                   uint32_t faceIndex,
                   const Library::Ptr& ftLibrary) :
    ft_(ftLibrary)
{
    if(FT_New_Face(*ft_, fontFilename.c_str(), faceIndex, &face_)) {
        std::ostringstream oss;
        oss << "rtac_display error : could not open font file :\n    "
            << fontFilename;
        throw std::runtime_error(oss.str());
    }
}

FontFace::Ptr FontFace::Create(const std::string& fontFilename,
                               uint32_t faceIndex,
                               Library::Ptr ftLibrary)
{
    if(!ftLibrary) {
        ftLibrary = Library::Create();
    }

    // This ensure proper reference counting see enable_shared_from_this
    // documentation for more info.
    auto tmp = Ptr(new FontFace(fontFilename, faceIndex, ftLibrary));
    return tmp->shared_from_this();
}

void FontFace::load_glyphs(FT_UInt pixelWidth, FT_UInt pixelHeight)
{
    if(FT_Set_Pixel_Sizes(face_, pixelWidth, pixelHeight)) {
        std::ostringstream oss;
        oss << "rtac_display error : Invalid text pixel size ("
            << pixelWidth << ", " << pixelHeight << ").";
        throw std::runtime_error(oss.str());
    }
    
    glyphs_.clear();
    for(uint8_t c = 0; c < 128; c++) {
        if(FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
            std::cerr << "rtac_display error : failed to load glyph '"
                      << c << "'" << std::endl;
        }
        std::cout << "Glyph " << c << ":\n" << face_->glyph->metrics << std::endl;
        std::cout << "bitmap : " << face_->glyph->bitmap.width << "x"
                                 << face_->glyph->bitmap.rows << std::endl << std::endl;
        glyphs_.emplace(std::make_pair(c, Glyph(face_)));
    }
}

const FontFace::GlyphMap& FontFace::glyphs() const
{
    return glyphs_;
}

const Glyph& FontFace::glyph(uint8_t c) const
{
    return glyphs_.at(c);
}

const FT_Face& FontFace::face() const
{
    return face_;
}

float FontFace::ascender() const
{
    return ((float)face_->ascender * face_->size->metrics.y_ppem) / face_->units_per_EM;
}

float FontFace::descender() const
{
    return ((float)face_->descender * face_->size->metrics.y_ppem) / face_->units_per_EM;
}

float FontFace::baselineskip() const
{
    return face_->size->metrics.height / 64.0f;
}

float FontFace::max_advance() const
{
    return face_->size->metrics.max_advance / 64.0f;
}

}; //namespace text
}; //namespace display
}; //namespace rtac

