#include <rtac_display/renderers/ImageRenderer.h>

namespace rtac { namespace display {

/**
 * This vertex shader expects coordinates of the corner of the image, and
 * automatically generates texture coordinates accordingly.
 */
const std::string ImageRenderer::vertexShader = std::string( R"(
#version 430 core

in vec2 point;
in vec2 uvIn;
out vec2 uv;
uniform mat4 view;

void main()
{
    //gl_Position = vec4(point, 0.0, 1.0);
    gl_Position = view*vec4(point, 0.0, 1.0);
    //uv = 0.5f*(point.xy + 1.0f);
    //uv.x = 0.5f*(point.x + 1.0f);
    //uv.y = 0.5f*(1.0f - point.y);
    uv = uvIn;
}
)");

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string ImageRenderer::fragmentShader = std::string(R"(
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
const std::string ImageRenderer::colormapFragmentShader = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;
uniform sampler2D colormap;

out vec4 outColor;

void main()
{
    outColor = texture(colormap, vec2(texture(tex, uv).x, 0.0));
}
)");

/**
 * Creates a new ImageRenderer object on the heap and outputs a shared_ptr.
 *
 * An OpenGL context must have been created beforehand.
 */
ImageRenderer::Ptr ImageRenderer::Create(const GLContext::Ptr& context)
{
    return Ptr(new ImageRenderer(context));
}

/**
 * ImageRenderer does not expects any parameters. It creates its own view.
 *
 * An OpenGL context must have been created before any instantiation.
 */
ImageRenderer::ImageRenderer(const GLContext::Ptr& context) :
    Renderer(context, vertexShader, fragmentShader),
    texture_(GLTexture::New()),
    imageView_(ImageView::New()),
    passThroughProgram_(this->renderProgram_),
    colormapProgram_(create_render_program(vertexShader, colormapFragmentShader)),
    verticalFlip_(true) // More natural for CPU texture
{}

GLTexture::Ptr& ImageRenderer::texture()
{
    return texture_;
}

GLTexture::ConstPtr ImageRenderer::texture() const
{
    return texture_;
}

void ImageRenderer::set_colormap(const Colormap::Ptr& colormap)
{
    colormap_ = colormap;
    this->enable_colormap();
}

bool ImageRenderer::enable_colormap()
{
    if(!colormap_) {
        this->set_viridis_colormap();
    }
    renderProgram_ = colormapProgram_;
    return true;
}

void ImageRenderer::disable_colormap()
{
    renderProgram_ = passThroughProgram_;
}

bool ImageRenderer::uses_colormap() const
{
    return renderProgram_ == colormapProgram_;
}

void ImageRenderer::set_vertical_flip(bool doFlip)
{
    verticalFlip_ = doFlip;
}

void ImageRenderer::set_viridis_colormap()
{
    this->set_colormap(colormap::Viridis());
}

void ImageRenderer::set_gray_colormap()
{
    this->set_colormap(colormap::Gray());
}



/**
 * Generate screen coordinates of corner of image and displays the image.
 */
void ImageRenderer::draw(const View::ConstPtr& view) const
{
    imageView_->set_screen_size(view->screen_size());
    imageView_->set_image_shape(texture_->shape());

    static const float uvNoFlip[] = {0.0, 0.0,
                                     1.0, 0.0,
                                     1.0, 1.0,
                                     0.0, 1.0};
    static const float uvFlip[] = {0.0, 1.0,
                                   1.0, 1.0,
                                   1.0, 0.0,
                                   0.0, 0.0};
    unsigned int indexes[] = {0, 1, 2,
                              0, 2, 3};
    std::vector<float> vertices(8);
    vertices[0] = 0.0f;                    vertices[1] = 0.0f;
    vertices[2] = texture_->shape().width; vertices[3] = 0.0f;
    vertices[4] = texture_->shape().width; vertices[5] = texture_->shape().height;
    vertices[6] = 0.0f;                    vertices[7] = texture_->shape().height;
    


    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices.data());
    glEnableVertexAttribArray(0);
    if(verticalFlip_)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uvNoFlip);
    else
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uvFlip);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, imageView_->view_matrix().data());


    glUniform1i(glGetUniformLocation(renderProgram_, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());
    
    if(this->uses_colormap()) {
        glUniform1i(glGetUniformLocation(renderProgram_, "colormap"), 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, colormap_->texture().gl_id());
    }
     
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

}; //namespace display
}; //namespace rtac

