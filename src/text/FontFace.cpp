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
    return Ptr(new FontFace(fontFilename, faceIndex, ftLibrary));
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
        glyphs_.emplace(std::make_pair(c, Glyph(face_)));
    }
}

}; //namespace text
}; //namespace display
}; //namespace rtac

