#include <rtac_display/text/TextRenderer.h>

namespace rtac { namespace display { namespace text {

TextRenderer::TextRenderer(const FontFace::ConstPtr& font) :
    font_(font)
{
    if(!font_) {
        std::ostringstream oss;
        oss << "Error rtac_display::text::TextRenderer : "
            << "Invalid font face pointer.";
        throw std::runtime_error(oss.str());
    }
}

TextRenderer::Ptr TextRenderer::Create(const FontFace::ConstPtr& font,
                                       const std::string& text)
{
    auto renderer = Ptr(new TextRenderer(font));
    renderer->set_text(text);
    return renderer;
}

void TextRenderer::set_text(const std::string& text)
{
    text_ = text;
    this->update_texture();
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
                  (size_t)(lineCount * font_->baselineskip() + 1)});
}

void TextRenderer::update_texture()
{
    Shape textArea = this->compute_text_area(text_);
    std::cout << "Text area size : " << textArea << std::endl;
    texture_.set_size<types::Point4<float>>(textArea);

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

    glClearColor(0.0,0.0,0.0,0.0);
    //glClearColor(0.0,1.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, texture_.shape().width, texture_.shape().height);
    //glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

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
        
        std::cout << "character : " << c
                  << ", shape : " << glyph->shape()
                  << ", texture shape : " << glyph->texture().shape() << std::endl;
        glyph->draw(origin * current);
        current(0,3) += glyph->advance().x;
    }

    // unbinding the frame buffer for re-enabling on-screen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
    view(0,0) = 2.0f / texture_.width();
    view(1,1) = -2.0f / texture_.height();

    view(0,3) = -1.0f;
    view(1,3) = -(texture_.height() - 2.0f*font_->ascender()) / texture_.height();

    return view;
}

void TextRenderer::draw()
{
}

}; //namespace text
}; //namespace display
}; //namespace rtac


