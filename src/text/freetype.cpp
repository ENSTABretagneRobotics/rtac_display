#include <rtac_display/text/freetype.h>

namespace rtac { namespace display { namespace text {

Library::Library()
{
    if(FT_Init_FreeType(&library_)) {
        throw std::runtime_error("Could not initialize FreeType");
    }
}

Library::~Library()
{
    if(FT_Done_FreeType(library_)) {
        std::cerr << "Warning : error deallocating FreeType library" << std::endl; 
    }
}

Library::Ptr Library::Create()
{
    return Ptr(new Library);
}

Library::operator FT_Library()
{
    return library_;
}

}; //namespace text
}; //namespace display
}; //namespace rtac
