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
const std::string Glyph::fragmentShaderFlat = std::string(R"(
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

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string Glyph::fragmentShaderSubPix = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;
uniform vec3 color;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(tex, uv);
}
)");


Glyph::Glyph(FT_GlyphSlot glyph) :
    bearing_({(float)glyph->bitmap_left,
              (float)glyph->bitmap_top}),
    //bearing_({glyph->metrics.horiBearingX / 64.0f,
    //          glyph->metrics.horiBearingY / 64.0f}),
    advance_({glyph->advance.x / 64.0f,
              glyph->advance.y / 64.0f}),
    shape_({glyph->metrics.width  / 64.0f,
            glyph->metrics.height / 64.0f}),
    renderProgramFlat_(create_render_program(vertexShader, fragmentShaderFlat)),
    renderProgramSubPix_(create_render_program(vertexShader, fragmentShaderSubPix)),
    renderProgram_(renderProgramFlat_)
{
    this->load_bitmap(glyph);
}

void Glyph::load_bitmap(FT_GlyphSlot glyph)
{
    switch(glyph->bitmap.pixel_mode) {
        default: {
            std::ostringstream oss;
            oss << "Glyph::load_bitmap error : pixel type "
                << glyph->bitmap.pixel_mode << " not implemented.";
            throw std::runtime_error(oss.str());
            }
            break;
        case FT_PIXEL_MODE_GRAY:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            texture_.set_image({glyph->bitmap.width,
                                glyph->bitmap.rows},
                                (const uint8_t*)glyph->bitmap.buffer);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            renderProgram_ = renderProgramFlat_;
            break;
        case FT_PIXEL_MODE_LCD: {
            unsigned int W = glyph->bitmap.width / 3;
            unsigned int H = glyph->bitmap.rows;
            std::vector<Color::RGBA8> data(W*H);
            //std::vector<Color::RGB8> data(W*H);
            auto itIn  = glyph->bitmap.buffer;
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    data[W*h + w].r = itIn[3*w];
                    data[W*h + w].g = itIn[3*w + 1];
                    data[W*h + w].b = itIn[3*w + 2];
                    data[W*h + w].a = 255;
                }
                itIn += glyph->bitmap.pitch;
            }
            texture_.set_image({W,H}, data.data());

            //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            // texture_.set_size<Color::RGB8>({W,H});
            // glBindTexture(GL_TEXTURE_2D, texture_.gl_id());
            // // SRGB not handled for now in GLTexture
            // glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, W, H,
            //     0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
            // GL_CHECK_LAST();
            // glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            // glBindTexture(GL_TEXTURE_2D, 0);

            renderProgram_ = renderProgramSubPix_;
            }
            break;
    }

    texture_.bind(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

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
void Glyph::draw(const Mat4& view, const Color::RGBAf& color) const
{
    using Point2f = types::Point2<float>;
    //std::array<Point2f,4> points = {
    //    Point2f({bearing_.x,            bearing_.y - shape_.y}),
    //    Point2f({bearing_.x + shape_.x, bearing_.y - shape_.y}),
    //    Point2f({bearing_.x + shape_.x, bearing_.y}),
    //    Point2f({bearing_.x           , bearing_.y})};
    std::array<Point2f,4> points = {
        Point2f({bearing_.x,                    bearing_.y - texture_.height()}),
        Point2f({bearing_.x + texture_.width(), bearing_.y - texture_.height()}),
        Point2f({bearing_.x + texture_.width(), bearing_.y}),
        Point2f({bearing_.x,                    bearing_.y})};
    // Inverting texture coordinates for rendering (OpenGL texture origin is
    // lower-left corner, but everything else is usually upper-left).
    static const std::array<Point2f,4> uv[] = {Point2f({0.0f, 1.0f}),
                                               Point2f({1.0f, 1.0f}),
                                               Point2f({1.0f, 0.0f}),
                                               Point2f({0.0f, 0.0f})};
    static const unsigned int indexes[] = {0, 1, 2,
                                           0, 2, 3};

    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, points.data());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uv);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, view.data());
    glUniform3fv(glGetUniformLocation(renderProgram_, "color"), 1, (const float*)&color);

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

