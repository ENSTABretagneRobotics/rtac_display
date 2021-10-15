#include <iostream>
using namespace std;

#include <ft2build.h>
#include FT_FREETYPE_H

#include <rtac_base/files.h>
using namespace rtac;

int main()
{
    FT_Library ft;
    if(FT_Init_FreeType(&ft)) {
        throw std::runtime_error("Could not initialized FreeType");
    }

    FT_Face face;
    if(FT_New_Face(ft, "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 0, &face)) {
        throw std::runtime_error("Could not find font");
    }

    FT_Set_Pixel_Sizes(face, 0, 48);

    if(FT_Load_Char(face, 'X', FT_LOAD_RENDER)) {
        throw std::runtime_error("Failed to load glyph");
    }

    files::write_pgm("X.pgm", face->glyph->bitmap.width, face->glyph->bitmap.rows,
                     (const char*)face->glyph->bitmap.buffer);
    return 0;
}



