#include <rtac_display/text/Glyph.h>

namespace rtac { namespace display { namespace text {

/**
 * This vertex shader expects coordinates of the corner of the image, and
 * automatically generates texture coordinates accordingly.
 */
const std::string Glyph::vertexShader = std::string(R"(
#version 430 core

in vec2 point;
in vec2 uvIn;
out vec2 uv;

uniform mat4 view;

void main()
{
    gl_Position = view*vec4(point, 0.0, 1.0);
    uv = uvIn;
    //uv.x = 0.5f*(point.x + 1.0f);
    //uv.y = 0.5f*(1.0f - point.y);
}
)");

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string Glyph::fragmentShader = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;
uniform vec3 color;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(color, texture(tex, uv).x);
    //outColor = vec4(0.0,0.0,0.0,1.0);
}
)");

Glyph::Glyph(FT_Face face) :
    //bearing_({(float)face->glyph->bitmap_left,
    //          (float)face->glyph->bitmap_top}),
    bearing_({face->glyph->metrics.horiBearingX / 64.0f,
              face->glyph->metrics.horiBearingY / 64.0f}),
    advance_({face->glyph->advance.x / 64.0f,
              face->glyph->advance.y / 64.0f}),
    shape_({face->glyph->metrics.width  / 64.0f,
            face->glyph->metrics.height / 64.0f}),
    renderProgram_(create_render_program(vertexShader, fragmentShader))
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    texture_.set_image({face->glyph->bitmap.width,
                        face->glyph->bitmap.rows},
                        (const uint8_t*)face->glyph->bitmap.buffer);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    texture_.bind(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    texture_.unbind(GL_TEXTURE_2D);
}

Glyph::Glyph(Glyph&& other) :
    bearing_(std::move(other.bearing_)),
    advance_(std::move(other.advance_)),
    shape_  (std::move(other.shape_)),
    texture_(std::move(other.texture_)),
    renderProgram_(std::exchange(other.renderProgram_, 0))
{}

Glyph& Glyph::operator=(Glyph&& other)
{
    bearing_ = std::move(other.bearing_);
    advance_ = std::move(other.advance_);
    shape_   = std::move(other.shape_);
    texture_ = std::move(other.texture_);
    renderProgram_ = std::exchange(other.renderProgram_, 0);
    return *this;
}

types::Point2<float> Glyph::bearing() const
{
    return bearing_;
}

types::Point2<float> Glyph::advance() const
{
    return advance_;
}

const GLTexture& Glyph::texture() const
{
    return texture_;
}

types::Point2<float> Glyph::shape() const
{
    return shape_;
}

/**
 * Renders a glyph on the full viewport
 *
 * This function assumes that the viewport was set to the proper size and
 * position by another entity (namely a renderer holding the full sentence).
 */
void Glyph::draw(const Mat4& view, const std::array<float,3>& color) const
{
    using Point2f = types::Point2<float>;
    std::array<Point2f,4> points = {
        Point2f({bearing_.x,            bearing_.y - shape_.y}),
        Point2f({bearing_.x + shape_.x, bearing_.y - shape_.y}),
        Point2f({bearing_.x + shape_.x, bearing_.y}),
        Point2f({bearing_.x           , bearing_.y})};
    // Inverting texture coordinates for rendering (OpenGL texture origin is
    // lower-left corner, but everything else is usually upper-left).
    static const std::array<Point2f,4> uv[] = {Point2f({0.0f, 1.0f}),
                                               Point2f({1.0f, 1.0f}),
                                               Point2f({1.0f, 0.0f}),
                                               Point2f({0.0f, 0.0f})};
    static const unsigned int indexes[] = {0, 1, 2,
                                           0, 2, 3};

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, points.data());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uv);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, view.data());
    glUniform3fv(glGetUniformLocation(renderProgram_, "color"), 1, color.data());

    glUniform1i(glGetUniformLocation(renderProgram_, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    texture_.bind(GL_TEXTURE_2D);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);

    texture_.unbind(GL_TEXTURE_2D);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

}; //namespace text
}; //namespace display
}; //namespace rtac

