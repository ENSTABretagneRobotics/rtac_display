#include <rtac_display/text/TextRenderer.h>

namespace rtac { namespace display { namespace text {

/**
 * Point position computed on CPU side. 
 */
const std::string TextRenderer::vertexShader = std::string( R"(
#version 430 core

in vec4 point;
in vec2 uvIn;
out vec2 uv;

void main()
{
    gl_Position = point;
    uv = uvIn;
}
)");

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string TextRenderer::fragmentShaderFlat = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;

out vec4 outColor;

void main()
{
    outColor = texture(tex, uv);
}
)");

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string TextRenderer::fragmentShaderSubPix = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;
uniform vec4 color;

layout(location = 0, index = 0) out vec4 outColor;
layout(location = 0, index = 1) out vec4 outMask;

void main()
{
    outColor = color;
    outMask  = color.a*texture(tex, uv);
}
)");

TextRenderer::TextRenderer(const GLContext::Ptr& context,
                           const FontFace::ConstPtr& font) :
    Renderer(context, vertexShader, fragmentShaderFlat),
    font_(font),
    origin_({0,0,0,1}),
    anchor_({0,1.0f}),
    textColor_({0,0,0}),
    backColor_({0,0,0,0}),
    renderProgramFlat_(renderProgram_),
    renderProgramSubPix_(create_render_program(vertexShader, fragmentShaderSubPix))
{
    if(!font_) {
        std::ostringstream oss;
        oss << "Error rtac_display::text::TextRenderer : "
            << "Invalid font face pointer.";
        throw std::runtime_error(oss.str());
    }
}

TextRenderer::Ptr TextRenderer::Create(const GLContext::Ptr& context,
                                       const FontFace::ConstPtr& font,
                                       const std::string& text)
{
    auto renderer = Ptr(new TextRenderer(context, font));
    renderer->set_text(text);
    return renderer;
}

void TextRenderer::set_text(const std::string& text, bool updateNow)
{
    text_ = text;
    if(updateNow)
        this->update_texture();
}

void TextRenderer::set_text_color(const Color::RGBAf& color, bool updateNow)
{
    textColor_ = color;
    if(updateNow)
        this->update_texture();
}

void TextRenderer::set_back_color(const Color::RGBAf& color, bool updateNow)
{
    backColor_ = color;
    if(updateNow)
        this->update_texture();
}

void TextRenderer::set_anchor(const std::string& desc)
{
    if(desc.find("center") != std::string::npos) {
        anchor_(0) = 0.5f;
        anchor_(1) = 0.5f;
        return;
    }
    if(desc.find("left") != std::string::npos) {
        anchor_(0) = 0.0f;
    }
    else if(desc.find("right") != std::string::npos) {
        anchor_(0) = 1.0f;
    }
    if(desc.find("top") != std::string::npos) {
        anchor_(1) = 1.0f;
    }
    else if(desc.find("bottom") != std::string::npos) {
        anchor_(1) = 0.0f;
    }
}

Shape TextRenderer::compute_text_area(const std::string& text)
{
    if(!font_ || text.size() == 0)
        return Shape({0,0});

    int lineCount = 1;
    float maxWidth = 0.0f, currentWidth = 0.0f;
    for(auto c : text) {
        if(c == '\n') {
            lineCount++;
            maxWidth = std::max(maxWidth, currentWidth);
            currentWidth = 0.0f;
            continue;
        }
        if(c < 32 || c == 127) {
            // non-printable characters
            continue;
        }
        try {
            currentWidth += font_->glyph(c).advance().x;
        }
        catch(const std::out_of_range&) {
            currentWidth += font_->glyph('\n').advance().x;
        }
    }
    maxWidth = std::max(maxWidth, currentWidth);
    return Shape({(size_t)(4*(((int)maxWidth + 3) / 4)),
                  (size_t)(lineCount * font_->baselineskip())});
}

void TextRenderer::update_texture()
{
    Shape textArea = this->compute_text_area(text_);
    std::cout << "Text area size : " << textArea << std::endl;
    texture_.set_size<types::Point4<float>>(textArea);
    texture_.bind(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    texture_.unbind(GL_TEXTURE_2D);

    // Preparing a framebuffer for off-screen rendering
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_.gl_id(), 0);
    
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::ostringstream oss;
        oss << "TextRenderer error : something went wrong when creating a framebuffer "
            << "(GL error : 0x" << std::hex << glGetError() << ")";
        throw std::runtime_error(oss.str());
    }
    texture_.unbind(GL_TEXTURE_2D);

    glViewport(0, 0, texture_.shape().width, texture_.shape().height);
    glClearColor(backColor_.r, backColor_.g, backColor_.b, backColor_.a);
    glClear(GL_COLOR_BUFFER_BIT);

    Mat4 origin  = this->view_matrix();
    Mat4 current = Mat4::Identity();

    const Glyph* glyph = nullptr;
    for(auto c : text_) {
        if(c == '\n') {
            current(1,3) -= font_->baselineskip();
            current(0,3)  = 0.0f;
            continue;
        }
        if(c < 32 || c == 127) {
            continue;
        }
        try {
            glyph = &font_->glyph(c);
        }
        catch(const std::out_of_range&) {
            // fallback to another glyph if not available
            glyph = &font_->glyph('\n');
        }
        
        //std::cout << "character : " << c
        //          << ", shape : " << glyph->shape()
        //          << ", texture shape : " << glyph->texture().shape() << std::endl;
        glyph->draw(origin * current, textColor_);
        current(0,3) += glyph->advance().x;
    }

    // unbinding the frame buffer for re-enabling on-screen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    switch(font_->render_mode()) {
        default: {
            std::ostringstream oss;
            oss << "TextRenderer::update_texture : pixel type "
                << font_->render_mode() << " not implemented.";
            throw std::runtime_error(oss.str());
            }
            break;
        case FT_RENDER_MODE_NORMAL:
            renderProgram_ = renderProgramFlat_;
            break;
        case FT_RENDER_MODE_LCD:
            renderProgram_ = renderProgramSubPix_;
            break;
    }
    //glEnable(GL_FRAMEBUFFER_SRGB);

    GL_CHECK_LAST();
}

const std::string& TextRenderer::text() const
{
    return text_;
}

FontFace::ConstPtr TextRenderer::font() const
{
    return font_;
}

const GLTexture& TextRenderer::texture() const
{
    return texture_;
}

/**
 * Return a view matrix transforming from 2D pixel space of the text areato
 * OpenGL clip space [-1,1] with origin at the left end of the first baseline.
 */ 
TextRenderer::Mat4 TextRenderer::view_matrix() const
{
    Mat4 view = Mat4::Identity();
    //view(0,0) =  2.0f / (texture_.width()  - 1);
    //view(1,1) = -2.0f / (texture_.height() - 1);
    view(0,0) =  2.0f / texture_.width();
    view(1,1) = -2.0f / texture_.height();

    view(0,3) = -1.0f;
    view(1,3) = -(texture_.height() - 2.0f*font_->ascender()) / texture_.height();

    return view;
}

TextRenderer::Vec4& TextRenderer::origin()
{
    return origin_;
}

const TextRenderer::Vec4& TextRenderer::origin() const
{
    return origin_;
}

TextRenderer::Vec2& TextRenderer::anchor()
{
    return anchor_;
}

const TextRenderer::Vec2& TextRenderer::anchor() const
{
    return anchor_;
}

const Color::RGBAf& TextRenderer::text_color() const
{
    return textColor_;
}

const Color::RGBAf& TextRenderer::back_color() const
{
    return backColor_;
}

float TextRenderer::anchor_depth(const View::ConstPtr& view) const
{
    Vec4 clipOrigin = view->view_matrix() * origin_;
    return clipOrigin(2) / clipOrigin(3);
}

std::array<TextRenderer::Vec4,4> TextRenderer::compute_corners(const View::ConstPtr& view) const
{
    float clipWidth  = (2.0f*texture_.width() ) / view->screen_size().width;
    float clipHeight = (2.0f*texture_.height()) / view->screen_size().height;
    //float clipWidth  = (2.0f*texture_.width() ) / (view->screen_size().width - 1);
    //float clipHeight = (2.0f*texture_.height()) / (view->screen_size().height- 1);

    // OpenGL clip space origin.
    Vec4 clipOrigin = view->view_matrix() * origin_;

    // Anchor shift
    clipOrigin(0) -= anchor_(0)*clipWidth;
    clipOrigin(1) -= anchor_(1)*clipHeight;

    // normalizing coordinates
    clipOrigin(0) /= clipOrigin(3);
    clipOrigin(1) /= clipOrigin(3);
    clipOrigin(2) /= clipOrigin(3);
    clipOrigin(3)  = 1.0f;

    std::array<Vec4,4> corners({
        Vec4(clipOrigin + Vec4({0,0,0,0})),
        Vec4(clipOrigin + Vec4({clipWidth,0,0,0})),
        Vec4(clipOrigin + Vec4({clipWidth,clipHeight,0,0})),
        Vec4(clipOrigin + Vec4({0,clipHeight,0,0}))});

    return corners;
}

void TextRenderer::draw(const View::ConstPtr& view) const
{
    auto corners = compute_corners(view);
    static const float uv[] = {0,1,
                               1,1,
                               1,0,
                               0,0};
    static const unsigned int indexes[] = {0, 1, 2,
                                           0, 2, 3};

    //glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_BLEND);
    if(renderProgram_ == renderProgramFlat_)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    else
        glBlendFunc(GL_SRC1_COLOR, GL_ONE_MINUS_SRC1_COLOR);

    glUseProgram(renderProgram_);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, corners.data());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uv);
    glEnableVertexAttribArray(1);

    glUniform1i(glGetUniformLocation(renderProgram_, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_.gl_id());

    if(renderProgram_ == renderProgramSubPix_) {
        glUniform4fv(glGetUniformLocation(renderProgram_, "color"), 1, (const float*)&textColor_);
    }
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);
    glDisable(GL_BLEND);
    //glDisable(GL_FRAMEBUFFER_SRGB);

    GL_CHECK_LAST();
}

}; //namespace text
}; //namespace display
}; //namespace rtac


