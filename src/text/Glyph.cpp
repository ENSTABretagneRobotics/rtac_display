#include <rtac_display/text/Glyph.h>

namespace rtac { namespace display { namespace text {

Glyph::Glyph(FT_Face face) :
    bearing_({face->glyph->bitmap_left,
              face->glyph->bitmap_top}),
    advance_({face->glyph->advance.x,
              face->glyph->advance.y})
{
    texture_.set_image({face->glyph->bitmap.width,
                        face->glyph->bitmap.rows},
                        (const uint8_t*)face->glyph->bitmap.buffer);

    texture_.bind(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    texture_.unbind(GL_TEXTURE_2D);
}

types::Point2<long> Glyph::bearing() const
{
    return bearing_;
}

types::Point2<long> Glyph::advance() const
{
    return advance_;
}

const GLTexture& Glyph::texture() const
{
    return texture_;
}

}; //namespace text
}; //namespace display
}; //namespace rtac

