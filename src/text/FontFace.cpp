#include <rtac_display/text/FontFace.h>

namespace rtac { namespace display { namespace text {

FontFace::FontFace(const std::string& fontFilename,
                   uint32_t faceIndex,
                   const Library::Ptr& ftLibrary,
                   FT_Render_Mode renderMode) :
    ft_(ftLibrary),
    renderMode_(renderMode)
{
    if(FT_New_Face(*ft_, fontFilename.c_str(), faceIndex, &face_)) {
        std::ostringstream oss;
        oss << "rtac_display error : could not open font file :\n    "
            << fontFilename;
        throw std::runtime_error(oss.str());
    }
    std::cout << face_->num_fixed_sizes << " fixed sizes." << std::endl;
    for(int i = 0; i < face_->num_fixed_sizes; i++) {
        std::cout << face_->available_sizes[i] << std::endl;
    }
}

FontFace::Ptr FontFace::Create(const std::string& fontFilename,
                               uint32_t faceIndex,
                               Library::Ptr ftLibrary,
                               FT_Render_Mode renderMode)
{
    if(!ftLibrary) {
        ftLibrary = Library::Create();
    }

    // This ensure proper reference counting see enable_shared_from_this
    // documentation for more info.
    auto tmp = Ptr(new FontFace(fontFilename, faceIndex, ftLibrary, renderMode));
    return tmp->shared_from_this();
}

void FontFace::set_char_size(float pt, FT_UInt screenDpi)
{
    if(FT_Set_Char_Size(face_, 0, (unsigned int) (pt * 64), screenDpi, screenDpi)) {
        std::ostringstream oss;
        oss << "rtac_display error : Invalid text char size (pt:"
            << pt << ", dpi:" << screenDpi << ")";
        throw std::runtime_error(oss.str());
    }
    this->load_glyphs();
}

    
void FontFace::set_pixel_size(FT_UInt size)
{
    if(FT_Set_Pixel_Sizes(face_, size, size)) {
        std::ostringstream oss;
        oss << "rtac_display error : Invalid text pixel size ("
            << size << ")";
        throw std::runtime_error(oss.str());
    }
    this->load_glyphs();
}

void FontFace::load_glyphs()
{
    glyphs_.clear();
    if(FT_Library_SetLcdFilter(*ft_, FT_LCD_FILTER_DEFAULT)) {
        throw std::runtime_error("Subpixel rendering is disabled");
    }
    for(uint8_t c = 0; c < 128; c++) {
        //if(FT_Load_Char(face_, c, FT_LOAD_DEFAULT)) {
        if(FT_Load_Char(face_, c, FT_LOAD_FORCE_AUTOHINT)) {
            std::cerr << "rtac_display error : failed to load glyph '"
                      << c << "'" << std::endl;
        }
        if(FT_Render_Glyph(face_->glyph, renderMode_)) {
            std::cerr << "rtac_display error : failed to render glyph '"
                      << c << "'" << std::endl;
        }

        glyphs_.emplace(std::make_pair(c, Glyph(face_->glyph)));
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
    // return face_->size->metrics.height / 64.0f;
    //return face_->height / 64.0f;
    // this aligns with ubuntu terminal (is it correct ?)
    return face_->size->metrics.height / 64.0f + 1.0f;
}

float FontFace::max_advance() const
{
    return face_->size->metrics.max_advance / 64.0f;
}

FT_Render_Mode FontFace::render_mode() const
{
    return renderMode_;
}

FT_Vector FontFace::get_kerning(char left, char right) const
{
    FT_Vector kerning({0,0});

    auto leftIndex  = FT_Get_Char_Index(face_, left);
    auto rightIndex = FT_Get_Char_Index(face_, right);

    if(!leftIndex || !rightIndex) {
        return kerning;
    }

    if(FT_Get_Kerning(face_, leftIndex, rightIndex,
                      FT_KERNING_DEFAULT, &kerning)) {
        std::cerr << "Unable to get kerning info for ("
                  << left << "," << right << ")" << std::endl;
        return kerning;
    }

    return kerning;
}

}; //namespace text
}; //namespace display
}; //namespace rtac

